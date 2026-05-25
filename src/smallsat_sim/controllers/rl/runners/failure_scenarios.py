from __future__ import annotations

import csv
import hashlib
import os
from functools import lru_cache
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np

from smallsat_sim.envs.perturbation_gp import ThrusterFailureSimulator
from smallsat_sim.envs.perturbation_state import (
    PerturbationState,
    PerturbationStatus,
)
from smallsat_sim.envs.perturbations_rl import Perturbation


SPLIT_TRAIN = 0
SPLIT_EVAL_ID = 1
SPLIT_EVAL_OOD = 2
SPLIT_STRESS = 3
BIN_EASY = 0
BIN_MEDIUM = 1
BIN_HARD = 2
REGIME_REDUNDANT = 0
REGIME_MARGINAL = 1
REGIME_AUTHORITY_LIMITED = 2
REGIME_BIAS_LIMITED = 3
FAILURE_TYPE_NAMES = {
    0: "stuck_off",
    1: "stuck_on",
    2: "faulty_valve",
    3: "saturated_thrust",
    4: "thrust_instability",
}
SPLIT_NAMES = {
    SPLIT_TRAIN: "train",
    SPLIT_EVAL_ID: "eval_id",
    SPLIT_EVAL_OOD: "eval_ood",
    SPLIT_STRESS: "stress",
}
BIN_NAMES = {
    BIN_EASY: "easy",
    BIN_MEDIUM: "medium",
    BIN_HARD: "hard",
}
AUTHORITY_REGIME_NAMES = {
    REGIME_REDUNDANT: "redundant",
    REGIME_MARGINAL: "marginal",
    REGIME_AUTHORITY_LIMITED: "authority_limited",
    REGIME_BIAS_LIMITED: "bias_limited",
}
SCENARIO_FAILURE_STATUS = {
    0: PerturbationStatus.STUCK_OFF.value,
    1: PerturbationStatus.STUCK_ON.value,
    2: PerturbationStatus.FAULTY_VALVE.value,
    3: PerturbationStatus.SATURATED_THRUST.value,
    4: PerturbationStatus.THRUST_INSTABILITY.value,
}
_GP_SAMPLE_BANK: dict[tuple, tuple[jnp.ndarray, jnp.ndarray]] = {}
_SCENARIO_CACHE_VERSION = "full-pose-authority-target-v1"


def _scenario_cache_dir() -> str:
    cache_dir = os.environ.get(
        "SMALLSAT_SCENARIO_CACHE_DIR",
        os.path.join(
            os.getcwd(),
            "src",
            "smallsat_sim",
            "controllers",
            "rl",
            "checkpoints",
            "scenario_cache",
        ),
    )
    os.makedirs(cache_dir, exist_ok=True)
    return cache_dir


def _scenario_cache_key(
    mixer_t: np.ndarray,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
    *,
    max_faults: int,
    min_rank: int,
    stress_quantile: float,
    mild_effectiveness: float,
) -> str:
    digest = hashlib.sha256()
    digest.update(_SCENARIO_CACHE_VERSION.encode())
    digest.update(np.asarray(mixer_t, dtype=np.float32).tobytes())
    digest.update(np.asarray(ctrl_low, dtype=np.float32).tobytes())
    digest.update(np.asarray(ctrl_high, dtype=np.float32).tobytes())
    digest.update(str(int(max_faults)).encode())
    digest.update(str(int(min_rank)).encode())
    digest.update(f"{float(stress_quantile):.8f}".encode())
    digest.update(f"{float(mild_effectiveness):.8f}".encode())
    return digest.hexdigest()[:16]


def _task_wrench_samples(mixer_t: np.ndarray, ctrl_high: np.ndarray) -> np.ndarray:
    nominal_map = mixer_t.T
    positive = np.sum(np.maximum(nominal_map, 0.0) * ctrl_high[None, :], axis=1)
    negative = np.sum(np.minimum(nominal_map, 0.0) * ctrl_high[None, :], axis=1)
    axis_mag = 0.35 * np.minimum(np.abs(positive), np.abs(negative))
    axis_mag = np.maximum(axis_mag, 1e-3)

    samples: list[np.ndarray] = []
    for axis in range(nominal_map.shape[0]):
        unit = np.zeros((nominal_map.shape[0],), dtype=np.float32)
        unit[axis] = axis_mag[axis]
        samples.append(unit.copy())
        samples.append(-unit.copy())
    for first in range(nominal_map.shape[0]):
        second = (first + 1) % nominal_map.shape[0]
        wrench = np.zeros((nominal_map.shape[0],), dtype=np.float32)
        wrench[first] = 0.5 * axis_mag[first]
        wrench[second] = 0.5 * axis_mag[second]
        samples.append(wrench)
        samples.append(-wrench)
    return np.asarray(samples, dtype=np.float32)


def _targeted_task_wrench_samples(
    mixer_t: np.ndarray, ctrl_high: np.ndarray
) -> np.ndarray:
    nominal_map = mixer_t.T
    positive = np.sum(np.maximum(nominal_map, 0.0) * ctrl_high[None, :], axis=1)
    negative = np.sum(np.minimum(nominal_map, 0.0) * ctrl_high[None, :], axis=1)
    axis_mag = 0.35 * np.minimum(np.abs(positive), np.abs(negative))
    axis_mag = np.maximum(axis_mag, 1e-3)

    samples: list[np.ndarray] = []
    for axis in range(nominal_map.shape[0]):
        unit = np.zeros((nominal_map.shape[0],), dtype=np.float32)
        unit[axis] = axis_mag[axis]
        samples.append(unit.copy())
        samples.append(-unit.copy())
    for first, second in ((0, 1), (0, 2), (1, 2), (3, 4), (3, 5), (4, 5)):
        for sign_first, sign_second in ((1.0, 1.0), (1.0, -1.0), (-1.0, 1.0), (-1.0, -1.0)):
            wrench = np.zeros((nominal_map.shape[0],), dtype=np.float32)
            wrench[first] = sign_first * 0.5 * axis_mag[first]
            wrench[second] = sign_second * 0.5 * axis_mag[second]
            samples.append(wrench)
    return np.asarray(samples, dtype=np.float32)


def _projected_bounded_residual(
    wrench_map: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_wrench: np.ndarray,
    *,
    iterations: int = 32,
) -> float:
    u = np.clip(np.linalg.pinv(wrench_map) @ target_wrench, lower, upper)
    lipschitz = float(np.linalg.norm(wrench_map, ord=2) ** 2) + 1e-6
    step_size = 1.0 / lipschitz
    for _ in range(iterations):
        residual = wrench_map @ u - target_wrench
        grad = wrench_map.T @ residual
        u = np.clip(u - step_size * grad, lower, upper)
    return float(np.linalg.norm(wrench_map @ u - target_wrench))


def _max_feasible_wrench_scale(
    wrench_map: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_wrench: np.ndarray,
    *,
    tolerance: float = 0.05,
    high: float = 4.0,
    iterations: int = 7,
) -> float:
    """
    Approximate the largest scale alpha such that alpha * target_wrench remains
    feasible under actuator bounds.

    Feasibility is tested by projected bounded least squares. The returned value
    has a direct control interpretation: alpha=1 means the sampled task wrench is
    just feasible, alpha>1 means authority margin remains, and alpha<1 means even
    the nominal sampled task wrench is outside the damaged wrench set.
    """
    target_norm = float(np.linalg.norm(target_wrench)) + 1e-6

    def feasible(scale: float) -> bool:
        residual = _projected_bounded_residual(
            wrench_map,
            lower,
            upper,
            scale * target_wrench,
            iterations=16,
        )
        return residual / (scale * target_norm + 1e-6) <= tolerance

    lo = 0.0
    hi = float(high)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        if feasible(mid):
            lo = mid
        else:
            hi = mid
    return lo


def _scenario_bounds(
    failure_types: tuple[int, ...],
    thrusters: tuple[int, ...],
    mixer_t: np.ndarray,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
    mild_effectiveness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    effectiveness = np.ones((mixer_t.shape[0],), dtype=np.float32)
    lower = ctrl_low.copy()
    upper = ctrl_high.copy()
    for failure_type, thruster in zip(failure_types, thrusters, strict=True):
        if failure_type == 0:
            effectiveness[thruster] = 0.0
            lower[thruster] = 0.0
            upper[thruster] = 0.0
        elif failure_type == 1:
            stuck_value = 0.5 * ctrl_high[thruster]
            lower[thruster] = stuck_value
            upper[thruster] = stuck_value
            effectiveness[thruster] = 1.0
        else:
            effectiveness[thruster] = min(effectiveness[thruster], mild_effectiveness)
            upper[thruster] = min(upper[thruster], mild_effectiveness * ctrl_high[thruster])
    return effectiveness, lower, upper


def _scenario_metrics(
    failure_types: tuple[int, ...],
    thrusters: tuple[int, ...],
    mixer_t: np.ndarray,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
    task_wrenches: np.ndarray,
    targeted_task_wrenches: np.ndarray,
    mild_effectiveness: float,
) -> tuple[
    int,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    float,
    int,
    float,
    float,
    np.ndarray,
]:
    effectiveness, lower, upper = _scenario_bounds(
        failure_types,
        thrusters,
        mixer_t,
        ctrl_low,
        ctrl_high,
        mild_effectiveness,
    )
    wrench_map = (effectiveness[:, None] * mixer_t).T
    singular_values = np.linalg.svd(wrench_map, compute_uv=False)
    rank = int(np.linalg.matrix_rank(wrench_map, tol=1e-6))
    min_sv = float(singular_values[-1]) if singular_values.size else 0.0
    condition_number = float(
        singular_values[0] / max(singular_values[-1], 1e-8)
    ) if singular_values.size else float("inf")

    errors = []
    margins = []
    for wrench in task_wrenches:
        residual = _projected_bounded_residual(wrench_map, lower, upper, wrench)
        errors.append(residual / (float(np.linalg.norm(wrench)) + 1e-6))
        feasible_scale = _max_feasible_wrench_scale(
            wrench_map, lower, upper, wrench
        )
        margins.append(feasible_scale - 1.0)
    errors_np = np.asarray(errors, dtype=np.float32)
    margins_np = np.asarray(margins, dtype=np.float32)
    targeted_errors = []
    targeted_margins = []
    for wrench in targeted_task_wrenches:
        residual = _projected_bounded_residual(wrench_map, lower, upper, wrench)
        targeted_errors.append(residual / (float(np.linalg.norm(wrench)) + 1e-6))
        feasible_scale = _max_feasible_wrench_scale(
            wrench_map, lower, upper, wrench
        )
        targeted_margins.append(feasible_scale - 1.0)
    targeted_errors_np = np.asarray(targeted_errors, dtype=np.float32)
    targeted_margins_np = np.asarray(targeted_margins, dtype=np.float32)
    target_score = targeted_errors_np - 0.05 * targeted_margins_np
    target_idx = int(np.argmax(target_score))
    targeted_wrench = targeted_task_wrenches[target_idx].astype(np.float32)
    zero_residual = _projected_bounded_residual(
        wrench_map, lower, upper, np.zeros((wrench_map.shape[0],), dtype=np.float32)
    )
    nominal_scale = float(np.median(np.linalg.norm(task_wrenches, axis=1))) + 1e-6
    bias_cancellation_error = zero_residual / nominal_scale
    bias_wrench = wrench_map @ np.clip(np.zeros_like(lower), lower, upper)
    bias_wrench_norm = float(np.linalg.norm(bias_wrench))
    p10_margin = float(np.percentile(margins_np, 10.0))
    mean_margin = float(np.mean(margins_np))
    p90_error = float(np.percentile(errors_np, 90.0))
    if bias_cancellation_error > 0.10:
        authority_regime = REGIME_BIAS_LIMITED
    elif p90_error > 0.10 or p10_margin < 0.05:
        authority_regime = REGIME_AUTHORITY_LIMITED
    elif p10_margin < 0.50:
        authority_regime = REGIME_MARGINAL
    else:
        authority_regime = REGIME_REDUNDANT
    return (
        rank,
        min_sv,
        condition_number,
        float(np.mean(errors_np)),
        p90_error,
        float(bias_cancellation_error),
        bias_wrench_norm,
        p10_margin,
        mean_margin,
        authority_regime,
        float(targeted_errors_np[target_idx]),
        float(targeted_margins_np[target_idx]),
        targeted_wrench,
    )


@lru_cache(maxsize=16)
def _build_scenario_table_cached(
    mixer_t_bytes: bytes,
    mixer_t_shape: tuple[int, int],
    ctrl_low_bytes: bytes,
    ctrl_high_bytes: bytes,
    max_faults: int,
    min_rank: int,
    stress_quantile: float,
    mild_effectiveness: float,
) -> dict[str, np.ndarray]:
    mixer_t = np.frombuffer(mixer_t_bytes, dtype=np.float32).reshape(mixer_t_shape)
    ctrl_low = np.frombuffer(ctrl_low_bytes, dtype=np.float32).copy()
    ctrl_high = np.frombuffer(ctrl_high_bytes, dtype=np.float32).copy()
    task_wrenches = _task_wrench_samples(mixer_t, ctrl_high)
    targeted_task_wrenches = _targeted_task_wrench_samples(mixer_t, ctrl_high)
    n_thrusters = mixer_t.shape[0]
    base_faults = [(failure_type, thruster) for failure_type in range(5) for thruster in range(n_thrusters)]

    scenario_rows = []
    p90_all = []
    train_candidate_indices = []
    deterministic_holdout = []

    for n_faults in range(1, max_faults + 1):
        for combo in combinations(base_faults, n_faults):
            thrusters = tuple(item[1] for item in combo)
            if len(set(thrusters)) != len(thrusters):
                continue
            failure_types = tuple(item[0] for item in combo)
            (
                rank,
                min_sv,
                condition_number,
                mean_error,
                p90_error,
                bias_cancellation_error,
                bias_wrench_norm,
                p10_authority_margin,
                mean_authority_margin,
                authority_regime,
                targeted_task_error,
                targeted_task_margin,
                targeted_task_wrench,
            ) = _scenario_metrics(
                failure_types,
                thrusters,
                mixer_t,
                ctrl_low,
                ctrl_high,
                task_wrenches,
                targeted_task_wrenches,
                mild_effectiveness,
            )
            if rank < min_rank:
                continue
            scenario_rows.append(
                {
                    "failure_types": failure_types,
                    "thrusters": thrusters,
                    "n_faults": n_faults,
                    "rank": rank,
                    "min_sv": min_sv,
                    "condition_number": condition_number,
                    "mean_error": mean_error,
                    "p90_error": p90_error,
                    "bias_cancellation_error": bias_cancellation_error,
                    "bias_wrench_norm": bias_wrench_norm,
                    "p10_authority_margin": p10_authority_margin,
                    "mean_authority_margin": mean_authority_margin,
                    "authority_regime": authority_regime,
                    "targeted_task_error": targeted_task_error,
                    "targeted_task_margin": targeted_task_margin,
                    "targeted_task_wrench": targeted_task_wrench,
                }
            )
            p90_all.append(p90_error)
            row_idx = len(scenario_rows) - 1
            holdout_hash = (
                sum((ft + 1) * 17 for ft in failure_types)
                + sum((thr + 1) * 31 for thr in thrusters)
                + n_faults * 13
            ) % 10
            deterministic_holdout.append(holdout_hash >= 8)
            if holdout_hash < 8:
                train_candidate_indices.append(row_idx)

    if not scenario_rows:
        empty_int = np.empty((0, max_faults), dtype=np.int32)
        return {
            "failure_types": empty_int,
            "thrusters": empty_int,
            "n_faults": np.empty((0,), dtype=np.int32),
            "split": np.empty((0,), dtype=np.int32),
            "difficulty_bin": np.empty((0,), dtype=np.int32),
            "rank": np.empty((0,), dtype=np.int32),
            "min_singular_value": np.empty((0,), dtype=np.float32),
            "condition_number": np.empty((0,), dtype=np.float32),
            "mean_feasibility_error": np.empty((0,), dtype=np.float32),
            "p90_feasibility_error": np.empty((0,), dtype=np.float32),
            "bias_cancellation_error": np.empty((0,), dtype=np.float32),
            "bias_wrench_norm": np.empty((0,), dtype=np.float32),
            "p10_authority_margin": np.empty((0,), dtype=np.float32),
            "mean_authority_margin": np.empty((0,), dtype=np.float32),
            "authority_regime": np.empty((0,), dtype=np.int32),
            "targeted_task_error": np.empty((0,), dtype=np.float32),
            "targeted_task_margin": np.empty((0,), dtype=np.float32),
            "targeted_task_wrench": np.empty((0, 6), dtype=np.float32),
        }

    p90_all_np = np.asarray(p90_all, dtype=np.float32)
    stress_threshold = float(np.quantile(p90_all_np, stress_quantile))
    train_p90 = np.asarray(
        [scenario_rows[idx]["p90_error"] for idx in train_candidate_indices],
        dtype=np.float32,
    )
    if train_p90.size == 0:
        train_p90 = p90_all_np
    easy_threshold = float(np.quantile(train_p90, 1.0 / 3.0))
    medium_threshold = float(np.quantile(train_p90, 2.0 / 3.0))

    failure_type_rows: list[list[int]] = []
    thruster_rows: list[list[int]] = []
    n_fault_rows: list[int] = []
    split_rows: list[int] = []
    rank_rows: list[int] = []
    min_sv_rows: list[float] = []
    condition_rows: list[float] = []
    difficulty_bin_rows: list[int] = []
    mean_error_rows: list[float] = []
    p90_error_rows: list[float] = []
    bias_cancel_rows: list[float] = []
    bias_norm_rows: list[float] = []
    p10_margin_rows: list[float] = []
    mean_margin_rows: list[float] = []
    authority_regime_rows: list[int] = []
    targeted_error_rows: list[float] = []
    targeted_margin_rows: list[float] = []
    targeted_wrench_rows: list[np.ndarray] = []

    for row_idx, row in enumerate(scenario_rows):
        failure_types = row["failure_types"]
        thrusters = row["thrusters"]
        n_faults = row["n_faults"]
        p90_error = row["p90_error"]
        if p90_error >= stress_threshold:
            split = SPLIT_STRESS
        elif deterministic_holdout[row_idx]:
            split = SPLIT_EVAL_OOD if n_faults > 1 else SPLIT_EVAL_ID
        else:
            split = SPLIT_TRAIN

        if p90_error <= easy_threshold:
            difficulty_bin = BIN_EASY
        elif p90_error <= medium_threshold:
            difficulty_bin = BIN_MEDIUM
        else:
            difficulty_bin = BIN_HARD

        padded_types = list(failure_types) + [-1] * (max_faults - n_faults)
        padded_thrusters = list(thrusters) + [-1] * (max_faults - n_faults)
        failure_type_rows.append(padded_types)
        thruster_rows.append(padded_thrusters)
        n_fault_rows.append(n_faults)
        split_rows.append(split)
        difficulty_bin_rows.append(difficulty_bin)
        rank_rows.append(row["rank"])
        min_sv_rows.append(row["min_sv"])
        condition_rows.append(row["condition_number"])
        mean_error_rows.append(row["mean_error"])
        p90_error_rows.append(p90_error)
        bias_cancel_rows.append(row["bias_cancellation_error"])
        bias_norm_rows.append(row["bias_wrench_norm"])
        p10_margin_rows.append(row["p10_authority_margin"])
        mean_margin_rows.append(row["mean_authority_margin"])
        authority_regime_rows.append(row["authority_regime"])
        targeted_error_rows.append(row["targeted_task_error"])
        targeted_margin_rows.append(row["targeted_task_margin"])
        targeted_wrench_rows.append(row["targeted_task_wrench"])

    return {
        "failure_types": np.asarray(failure_type_rows, dtype=np.int32),
        "thrusters": np.asarray(thruster_rows, dtype=np.int32),
        "n_faults": np.asarray(n_fault_rows, dtype=np.int32),
        "split": np.asarray(split_rows, dtype=np.int32),
        "difficulty_bin": np.asarray(difficulty_bin_rows, dtype=np.int32),
        "rank": np.asarray(rank_rows, dtype=np.int32),
        "min_singular_value": np.asarray(min_sv_rows, dtype=np.float32),
        "condition_number": np.asarray(condition_rows, dtype=np.float32),
        "mean_feasibility_error": np.asarray(mean_error_rows, dtype=np.float32),
        "p90_feasibility_error": np.asarray(p90_error_rows, dtype=np.float32),
        "bias_cancellation_error": np.asarray(bias_cancel_rows, dtype=np.float32),
        "bias_wrench_norm": np.asarray(bias_norm_rows, dtype=np.float32),
        "p10_authority_margin": np.asarray(p10_margin_rows, dtype=np.float32),
        "mean_authority_margin": np.asarray(mean_margin_rows, dtype=np.float32),
        "authority_regime": np.asarray(authority_regime_rows, dtype=np.int32),
        "targeted_task_error": np.asarray(targeted_error_rows, dtype=np.float32),
        "targeted_task_margin": np.asarray(
            targeted_margin_rows, dtype=np.float32
        ),
        "targeted_task_wrench": np.asarray(
            targeted_wrench_rows, dtype=np.float32
        ),
    }


def build_failure_scenario_table(
    thruster_mixer_t: jnp.ndarray,
    ctrl_low: jnp.ndarray,
    ctrl_high: jnp.ndarray,
    *,
    max_faults: int = 2,
    min_rank: int = 6,
    stress_quantile: float = 0.9,
    mild_effectiveness: float = 0.5,
) -> dict[str, jnp.ndarray]:
    mixer_t = np.asarray(jax.device_get(thruster_mixer_t), dtype=np.float32)
    ctrl_low_np = np.asarray(jax.device_get(ctrl_low), dtype=np.float32)
    ctrl_high_np = np.asarray(jax.device_get(ctrl_high), dtype=np.float32)
    cache_key = _scenario_cache_key(
        mixer_t,
        ctrl_low_np,
        ctrl_high_np,
        max_faults=max_faults,
        min_rank=min_rank,
        stress_quantile=stress_quantile,
        mild_effectiveness=mild_effectiveness,
    )
    cache_path = os.path.join(_scenario_cache_dir(), f"{cache_key}.npz")
    if os.path.isfile(cache_path):
        print(f"[Failure Scenarios] Loading cached scenario table: {cache_path}", flush=True)
        with np.load(cache_path) as cached:
            return {key: jnp.asarray(cached[key]) for key in cached.files}

    print(
        "[Failure Scenarios] Building scenario table "
        f"(max_faults={max_faults}, min_rank={min_rank}); this is cached after the first run.",
        flush=True,
    )
    table = _build_scenario_table_cached(
        mixer_t.tobytes(),
        tuple(mixer_t.shape),
        ctrl_low_np.tobytes(),
        ctrl_high_np.tobytes(),
        int(max_faults),
        int(min_rank),
        float(stress_quantile),
        float(mild_effectiveness),
    )
    np.savez_compressed(cache_path, **table)
    print(f"[Failure Scenarios] Cached scenario table: {cache_path}", flush=True)
    return {key: jnp.asarray(value) for key, value in table.items()}


def scenario_split_counts(table: dict[str, jnp.ndarray]) -> dict[str, int]:
    split = np.asarray(jax.device_get(table["split"]))
    return {
        "train": int((split == SPLIT_TRAIN).sum()),
        "eval_id": int((split == SPLIT_EVAL_ID).sum()),
        "eval_ood": int((split == SPLIT_EVAL_OOD).sum()),
        "stress": int((split == SPLIT_STRESS).sum()),
    }


def scenario_bin_counts(table: dict[str, jnp.ndarray], split_id: int = SPLIT_TRAIN) -> dict[str, int]:
    split = np.asarray(jax.device_get(table["split"]))
    bins = np.asarray(jax.device_get(table["difficulty_bin"]))
    split_mask = split == split_id
    return {
        "easy": int(np.logical_and(split_mask, bins == BIN_EASY).sum()),
        "medium": int(np.logical_and(split_mask, bins == BIN_MEDIUM).sum()),
        "hard": int(np.logical_and(split_mask, bins == BIN_HARD).sum()),
    }


def scenario_authority_regime_counts(
    table: dict[str, jnp.ndarray],
    split_id: int | None = None,
) -> dict[str, int]:
    regimes = np.asarray(jax.device_get(table["authority_regime"]))
    if split_id is None:
        mask = np.ones_like(regimes, dtype=bool)
    else:
        split = np.asarray(jax.device_get(table["split"]))
        mask = split == int(split_id)
    return {
        name: int(np.logical_and(mask, regimes == regime_id).sum())
        for regime_id, name in AUTHORITY_REGIME_NAMES.items()
    }


def save_scenario_table_csv(table: dict[str, jnp.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    failure_types = np.asarray(jax.device_get(table["failure_types"]))
    thrusters = np.asarray(jax.device_get(table["thrusters"]))
    n_faults = np.asarray(jax.device_get(table["n_faults"]))
    splits = np.asarray(jax.device_get(table["split"]))
    bins = np.asarray(jax.device_get(table["difficulty_bin"]))
    ranks = np.asarray(jax.device_get(table["rank"]))
    min_sv = np.asarray(jax.device_get(table["min_singular_value"]))
    cond = np.asarray(jax.device_get(table["condition_number"]))
    mean_err = np.asarray(jax.device_get(table["mean_feasibility_error"]))
    p90_err = np.asarray(jax.device_get(table["p90_feasibility_error"]))
    bias_cancel = np.asarray(jax.device_get(table["bias_cancellation_error"]))
    bias_norm = np.asarray(jax.device_get(table["bias_wrench_norm"]))
    p10_margin = np.asarray(jax.device_get(table["p10_authority_margin"]))
    mean_margin = np.asarray(jax.device_get(table["mean_authority_margin"]))
    authority_regime = np.asarray(jax.device_get(table["authority_regime"]))
    targeted_error = np.asarray(jax.device_get(table["targeted_task_error"]))
    targeted_margin = np.asarray(jax.device_get(table["targeted_task_margin"]))
    targeted_wrench = np.asarray(jax.device_get(table["targeted_task_wrench"]))

    max_faults = failure_types.shape[1] if failure_types.ndim == 2 else 0
    fieldnames = [
        "scenario_id",
        "split",
        "difficulty_bin",
        "n_faults",
        "rank",
        "min_singular_value",
        "condition_number",
        "mean_feasibility_error",
        "p90_feasibility_error",
        "bias_cancellation_error",
        "bias_wrench_norm",
        "p10_authority_margin",
        "mean_authority_margin",
        "authority_regime",
        "targeted_task_error",
        "targeted_task_margin",
    ]
    fieldnames.extend([f"targeted_wrench_{idx}" for idx in range(6)])
    for idx in range(max_faults):
        fieldnames.extend(
            [
                f"failure_type_{idx}",
                f"failure_type_name_{idx}",
                f"thruster_{idx}",
            ]
        )

    with open(path, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for scenario_id in range(failure_types.shape[0]):
            row = {
                "scenario_id": scenario_id,
                "split": SPLIT_NAMES.get(int(splits[scenario_id]), "unknown"),
                "difficulty_bin": BIN_NAMES.get(int(bins[scenario_id]), "unknown"),
                "n_faults": int(n_faults[scenario_id]),
                "rank": int(ranks[scenario_id]),
                "min_singular_value": float(min_sv[scenario_id]),
                "condition_number": float(cond[scenario_id]),
                "mean_feasibility_error": float(mean_err[scenario_id]),
                "p90_feasibility_error": float(p90_err[scenario_id]),
                "bias_cancellation_error": float(bias_cancel[scenario_id]),
                "bias_wrench_norm": float(bias_norm[scenario_id]),
                "p10_authority_margin": float(p10_margin[scenario_id]),
                "mean_authority_margin": float(mean_margin[scenario_id]),
                "authority_regime": AUTHORITY_REGIME_NAMES.get(
                    int(authority_regime[scenario_id]), "unknown"
                ),
                "targeted_task_error": float(targeted_error[scenario_id]),
                "targeted_task_margin": float(targeted_margin[scenario_id]),
            }
            for wrench_idx in range(6):
                row[f"targeted_wrench_{wrench_idx}"] = float(
                    targeted_wrench[scenario_id, wrench_idx]
                )
            for fault_idx in range(max_faults):
                failure_type = int(failure_types[scenario_id, fault_idx])
                row[f"failure_type_{fault_idx}"] = failure_type
                row[f"failure_type_name_{fault_idx}"] = FAILURE_TYPE_NAMES.get(
                    failure_type, "none"
                )
                row[f"thruster_{fault_idx}"] = int(
                    thrusters[scenario_id, fault_idx]
                )
            writer.writerow(row)


def scenario_selection_payload(
    table: dict[str, jnp.ndarray],
    scenario_indices: jnp.ndarray,
    *,
    prefix: str = "scenario",
) -> dict[str, float]:
    scenario_indices_np = np.asarray(jax.device_get(scenario_indices), dtype=np.int32)
    if scenario_indices_np.size == 0:
        return {
            f"{prefix}/num_selected": 0.0,
            f"{prefix}/bin_easy_fraction": 0.0,
            f"{prefix}/bin_medium_fraction": 0.0,
            f"{prefix}/bin_hard_fraction": 0.0,
        }

    bins = np.asarray(jax.device_get(table["difficulty_bin"]))[scenario_indices_np]
    splits = np.asarray(jax.device_get(table["split"]))[scenario_indices_np]
    n_faults = np.asarray(jax.device_get(table["n_faults"]))[scenario_indices_np]
    p90_error = np.asarray(jax.device_get(table["p90_feasibility_error"]))[
        scenario_indices_np
    ]
    mean_error = np.asarray(jax.device_get(table["mean_feasibility_error"]))[
        scenario_indices_np
    ]
    bias_error = np.asarray(jax.device_get(table["bias_cancellation_error"]))[
        scenario_indices_np
    ]
    bias_norm = np.asarray(jax.device_get(table["bias_wrench_norm"]))[
        scenario_indices_np
    ]
    p10_margin = np.asarray(jax.device_get(table["p10_authority_margin"]))[
        scenario_indices_np
    ]
    mean_margin = np.asarray(jax.device_get(table["mean_authority_margin"]))[
        scenario_indices_np
    ]
    targeted_error = np.asarray(jax.device_get(table["targeted_task_error"]))[
        scenario_indices_np
    ]
    targeted_margin = np.asarray(jax.device_get(table["targeted_task_margin"]))[
        scenario_indices_np
    ]
    regimes = np.asarray(jax.device_get(table["authority_regime"]))[
        scenario_indices_np
    ]
    denom = float(max(scenario_indices_np.size, 1))
    return {
        f"{prefix}/num_selected": float(scenario_indices_np.size),
        f"{prefix}/bin_easy_fraction": float((bins == BIN_EASY).sum() / denom),
        f"{prefix}/bin_medium_fraction": float((bins == BIN_MEDIUM).sum() / denom),
        f"{prefix}/bin_hard_fraction": float((bins == BIN_HARD).sum() / denom),
        f"{prefix}/split_train_fraction": float((splits == SPLIT_TRAIN).sum() / denom),
        f"{prefix}/split_eval_id_fraction": float(
            (splits == SPLIT_EVAL_ID).sum() / denom
        ),
        f"{prefix}/split_eval_ood_fraction": float(
            (splits == SPLIT_EVAL_OOD).sum() / denom
        ),
        f"{prefix}/split_stress_fraction": float((splits == SPLIT_STRESS).sum() / denom),
        f"{prefix}/mean_num_faults": float(n_faults.mean()),
        f"{prefix}/mean_p90_feasibility_error": float(p90_error.mean()),
        f"{prefix}/max_p90_feasibility_error": float(p90_error.max()),
        f"{prefix}/mean_feasibility_error": float(mean_error.mean()),
        f"{prefix}/mean_bias_cancellation_error": float(bias_error.mean()),
        f"{prefix}/mean_bias_wrench_norm": float(bias_norm.mean()),
        f"{prefix}/mean_p10_authority_margin": float(p10_margin.mean()),
        f"{prefix}/mean_authority_margin": float(mean_margin.mean()),
        f"{prefix}/mean_targeted_task_error": float(targeted_error.mean()),
        f"{prefix}/mean_targeted_task_margin": float(targeted_margin.mean()),
        f"{prefix}/regime_redundant_fraction": float(
            (regimes == REGIME_REDUNDANT).sum() / denom
        ),
        f"{prefix}/regime_marginal_fraction": float(
            (regimes == REGIME_MARGINAL).sum() / denom
        ),
        f"{prefix}/regime_authority_limited_fraction": float(
            (regimes == REGIME_AUTHORITY_LIMITED).sum() / denom
        ),
        f"{prefix}/regime_bias_limited_fraction": float(
            (regimes == REGIME_BIAS_LIMITED).sum() / denom
        ),
    }


def sample_scenario_indices(
    key: jnp.ndarray,
    table: dict[str, jnp.ndarray],
    *,
    split_id: int,
    count: int,
    difficulty_bin: int | None = None,
    authority_regime: int | None = None,
) -> jnp.ndarray:
    split = table["split"]
    mask = split == int(split_id)
    if difficulty_bin is not None:
        bin_mask = jnp.logical_and(mask, table["difficulty_bin"] == int(difficulty_bin))
        mask = jnp.where(jnp.any(bin_mask), bin_mask, mask)
    if authority_regime is not None:
        regime_mask = jnp.logical_and(
            mask, table["authority_regime"] == int(authority_regime)
        )
        mask = jnp.where(jnp.any(regime_mask), regime_mask, mask)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    sampled = jax.random.categorical(key, logits, shape=(count,))
    return sampled.astype(jnp.int32)


def _scenario_rows_for_indices(
    table: dict[str, jnp.ndarray],
    selected_envs: jnp.ndarray,
    scenario_indices: jnp.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected_envs_np = np.asarray(jax.device_get(selected_envs), dtype=np.int32)
    scenario_indices_np = np.asarray(jax.device_get(scenario_indices), dtype=np.int32)
    failure_types = np.asarray(jax.device_get(table["failure_types"]))[
        scenario_indices_np
    ]
    thrusters = np.asarray(jax.device_get(table["thrusters"]))[scenario_indices_np]

    env_rows = np.repeat(selected_envs_np[:, None], failure_types.shape[1], axis=1)
    valid = np.logical_and(failure_types >= 0, thrusters >= 0)
    flat_envs = env_rows[valid].astype(np.int32)
    flat_thrusters = thrusters[valid].astype(np.int32)
    flat_failure_types = failure_types[valid].astype(np.int32)
    return flat_envs, flat_thrusters, flat_failure_types


def _gp_samples_for_failure(perturbation, failure_status: int, key: jnp.ndarray):
    max_force = getattr(perturbation, "thruster_list", [None])[0].ctrlrange[-1]
    valve_min = 0.15 * max_force
    valve_max = 0.8 * max_force
    simulator_kwargs = perturbation._gp_simulator_kwargs(
        upper_bound=max_force,
        valve_min=valve_min,
        valve_max=valve_max,
    )
    cache_key = (
        int(failure_status),
        float(simulator_kwargs["upper_bound"]),
        float(valve_min),
        float(valve_max),
        int(simulator_kwargs["num_points"]),
        int(simulator_kwargs["subset_size"]),
    )
    cached = _GP_SAMPLE_BANK.get(cache_key)
    if cached is not None:
        return cached

    x_data, y_data = ThrusterFailureSimulator(**simulator_kwargs).generate_failure_data(
        key, PerturbationStatus(failure_status)
    )
    x_data = jnp.asarray(x_data)
    y_data = jnp.asarray(y_data)
    _GP_SAMPLE_BANK[cache_key] = (x_data, y_data)
    return x_data, y_data


def precompute_scenario_gp_samples(env, key: jnp.ndarray) -> None:
    """
    Populate the global GP sample bank used by scenario application.

    Scenario curricula resample failures many times. The nonlinear GP failure
    curves are independent of the specific env assignment, so generating them
    once avoids repeated Cholesky/sampling work during later curriculum epochs.
    """
    if not hasattr(env, "perturbations") or env.perturbations is None:
        return
    subkeys = jax.random.split(key, 4)
    for failure_type in (2, 3, 4):
        failure_status = SCENARIO_FAILURE_STATUS[failure_type]
        samples = _gp_samples_for_failure(
            env.perturbations.perturbations[failure_type],
            failure_status,
            subkeys[failure_type - 1],
        )
        jax.block_until_ready(samples[1])


def apply_failure_scenarios(
    env,
    *,
    key: jnp.ndarray,
    table: dict[str, jnp.ndarray],
    selected_envs: jnp.ndarray,
    scenario_indices: jnp.ndarray,
    start_time: float | None,
) -> None:
    if selected_envs.size == 0:
        return

    flat_envs_np, flat_thrusters_np, flat_failure_types_np = _scenario_rows_for_indices(
        table, selected_envs, scenario_indices
    )
    if flat_envs_np.size == 0:
        return

    flat_envs = jnp.asarray(flat_envs_np, dtype=jnp.int32)
    flat_thrusters = jnp.asarray(flat_thrusters_np, dtype=jnp.int32)
    flat_status = jnp.asarray(
        [SCENARIO_FAILURE_STATUS[int(ft)] for ft in flat_failure_types_np],
        dtype=jnp.int32,
    )

    # Update the shared env/thruster failure mask once. This avoids the slow
    # interactive registration path, which grouped scenarios by type/thruster and
    # repeatedly synchronized with the host.
    thruster_mask = jnp.asarray(Perturbation.thruster_mask)
    Perturbation.thruster_mask = thruster_mask.at[flat_envs, flat_thrusters].set(
        flat_status
    )

    start_time_value = jnp.asarray(
        0.0 if start_time is None else start_time, dtype=jnp.float32
    )
    subkeys = jax.random.split(key, 6)
    perturbations = env.perturbations.perturbations

    for failure_type, failure_status in SCENARIO_FAILURE_STATUS.items():
        type_mask_np = flat_failure_types_np == failure_type
        type_envs_np = flat_envs_np[type_mask_np]
        type_thrusters_np = flat_thrusters_np[type_mask_np]
        perturbation = perturbations[failure_type]
        base_start_times = getattr(
            perturbation,
            "start_times",
            jnp.zeros((env.num_envs, env.act_dim), dtype=jnp.float32),
        )
        start_times = jnp.zeros_like(base_start_times)
        if type_envs_np.size > 0:
            type_envs = jnp.asarray(type_envs_np, dtype=jnp.int32)
            type_thrusters = jnp.asarray(type_thrusters_np, dtype=jnp.int32)
            start_times = start_times.at[type_envs, type_thrusters].set(
                start_time_value.astype(start_times.dtype)
            )

        perturbation.start_times = start_times
        if failure_type == 0:
            perturbation.state = PerturbationState(
                rng=perturbation._key,
                thruster_mask=Perturbation.thruster_mask,
                failure_value=failure_status,
                start_times=start_times,
            )
        elif failure_type == 1:
            stuck_force = jnp.asarray(perturbation.stuck_on_force)
            if type_envs_np.size > 0:
                type_envs = jnp.asarray(type_envs_np, dtype=jnp.int32)
                type_thrusters = jnp.asarray(type_thrusters_np, dtype=jnp.int32)
                min_force = perturbation.min_thruster_force[type_thrusters]
                max_force = perturbation.max_thruster_force[type_thrusters]
                sampled_force = jax.random.uniform(
                    subkeys[1],
                    shape=(type_envs.shape[0],),
                    minval=min_force,
                    maxval=max_force,
                )
                stuck_force = stuck_force.at[type_envs, type_thrusters].set(
                    sampled_force
                )
            perturbation.stuck_on_force = stuck_force
            perturbation.state = PerturbationState(
                rng=perturbation._key,
                thruster_mask=Perturbation.thruster_mask,
                failure_value=failure_status,
                start_times=start_times,
                max_thruster_force=stuck_force,
            )
        else:
            x_data, y_data = _gp_samples_for_failure(
                perturbation, failure_status, subkeys[failure_type]
            )
            gp_x = jnp.tile(x_data[None, :], (env.act_dim, 1))
            gp_y = jnp.tile(y_data[None, :], (env.act_dim, 1))
            perturbation.state = PerturbationState(
                rng=perturbation._key,
                thruster_mask=Perturbation.thruster_mask,
                failure_value=failure_status,
                start_times=start_times,
                gp_x_samples=gp_x,
                gp_y_samples=gp_y,
            )

    env._refresh_effect_states()
    env._state = env._state.replace(perturbation_states=env.perturbation_states)


def apply_sampled_failure_scenario_split(
    env,
    *,
    key: jnp.ndarray,
    table: dict[str, jnp.ndarray],
    split_id: int,
    fraction_perturbed_envs: float,
    start_time: float | None,
    difficulty_bin: int | None = None,
    authority_regime: int | None = None,
) -> dict[str, float]:
    num_perturbed = int(env.num_envs * max(0.0, min(1.0, fraction_perturbed_envs)))
    if num_perturbed <= 0:
        return scenario_selection_payload(
            table,
            jnp.empty((0,), dtype=jnp.int32),
        )

    select_key, scenario_key, apply_key = jax.random.split(key, 3)
    has_perturbation, has_disturbance = env._get_active_failure_masks()
    clean_envs = jnp.logical_not(jnp.logical_or(has_perturbation, has_disturbance))
    disturbed_only_envs = jnp.logical_and(
        has_disturbance, jnp.logical_not(has_perturbation)
    )
    selected_envs = env._select_envs_with_priority(
        select_key,
        num_perturbed,
        [clean_envs, disturbed_only_envs, has_perturbation],
    )
    scenario_indices = sample_scenario_indices(
        scenario_key,
        table,
        split_id=split_id,
        count=num_perturbed,
        difficulty_bin=difficulty_bin,
        authority_regime=authority_regime,
    )
    apply_failure_scenarios(
        env,
        key=apply_key,
        table=table,
        selected_envs=selected_envs,
        scenario_indices=scenario_indices,
        start_time=start_time,
    )
    return scenario_selection_payload(table, scenario_indices)
