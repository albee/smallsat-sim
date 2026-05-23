from __future__ import annotations

import csv
from functools import lru_cache
from itertools import combinations

import jax
import jax.numpy as jnp
import numpy as np


SPLIT_TRAIN = 0
SPLIT_EVAL_ID = 1
SPLIT_EVAL_OOD = 2
SPLIT_STRESS = 3
BIN_EASY = 0
BIN_MEDIUM = 1
BIN_HARD = 2
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


def _projected_bounded_residual(
    wrench_map: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_wrench: np.ndarray,
    *,
    iterations: int = 64,
) -> float:
    u = np.clip(np.linalg.pinv(wrench_map) @ target_wrench, lower, upper)
    lipschitz = float(np.linalg.norm(wrench_map, ord=2) ** 2) + 1e-6
    step_size = 1.0 / lipschitz
    for _ in range(iterations):
        residual = wrench_map @ u - target_wrench
        grad = wrench_map.T @ residual
        u = np.clip(u - step_size * grad, lower, upper)
    return float(np.linalg.norm(wrench_map @ u - target_wrench))


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
    mild_effectiveness: float,
) -> tuple[int, float, float, float, float, float, float]:
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
    for wrench in task_wrenches:
        residual = _projected_bounded_residual(wrench_map, lower, upper, wrench)
        errors.append(residual / (float(np.linalg.norm(wrench)) + 1e-6))
    errors_np = np.asarray(errors, dtype=np.float32)
    zero_residual = _projected_bounded_residual(
        wrench_map, lower, upper, np.zeros((wrench_map.shape[0],), dtype=np.float32)
    )
    nominal_scale = float(np.median(np.linalg.norm(task_wrenches, axis=1))) + 1e-6
    bias_cancellation_error = zero_residual / nominal_scale
    bias_wrench = wrench_map @ np.clip(np.zeros_like(lower), lower, upper)
    bias_wrench_norm = float(np.linalg.norm(bias_wrench))
    return (
        rank,
        min_sv,
        condition_number,
        float(np.mean(errors_np)),
        float(np.percentile(errors_np, 90.0)),
        float(bias_cancellation_error),
        bias_wrench_norm,
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
            ) = _scenario_metrics(
                failure_types,
                thrusters,
                mixer_t,
                ctrl_low,
                ctrl_high,
                task_wrenches,
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


def save_scenario_table_csv(table: dict[str, jnp.ndarray], path: str) -> None:
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
    ]
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
            }
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


def sample_scenario_indices(
    key: jnp.ndarray,
    table: dict[str, jnp.ndarray],
    *,
    split_id: int,
    count: int,
    difficulty_bin: int | None = None,
) -> jnp.ndarray:
    split = table["split"]
    mask = split == int(split_id)
    if difficulty_bin is not None:
        bin_mask = jnp.logical_and(mask, table["difficulty_bin"] == int(difficulty_bin))
        mask = jnp.where(jnp.any(bin_mask), bin_mask, mask)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    sampled = jax.random.categorical(key, logits, shape=(count,))
    return sampled.astype(jnp.int32)


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

    selected_envs_np = np.asarray(jax.device_get(selected_envs), dtype=np.int32)
    scenario_indices_np = np.asarray(jax.device_get(scenario_indices), dtype=np.int32)
    failure_types = np.asarray(jax.device_get(table["failure_types"]))[scenario_indices_np]
    thrusters = np.asarray(jax.device_get(table["thrusters"]))[scenario_indices_np]
    max_faults = int(failure_types.shape[1])
    subkeys = jax.random.split(key, max_faults * 5 + 1)[1:]

    for fault_slot in range(max_faults):
        slot_types = failure_types[:, fault_slot]
        slot_thrusters = thrusters[:, fault_slot]
        valid = slot_types >= 0
        for failure_type in range(5):
            for thruster in range(env.act_dim):
                mask = np.logical_and(
                    valid,
                    np.logical_and(slot_types == failure_type, slot_thrusters == thruster),
                )
                envs_for_failure_np = selected_envs_np[mask]
                if envs_for_failure_np.size == 0:
                    continue
                envs_for_failure = jnp.asarray(envs_for_failure_np, dtype=jnp.int32)
                thrusters_for_failure = jnp.full(
                    (envs_for_failure.shape[0],), thruster, dtype=jnp.int32
                )
                failure_key = subkeys[fault_slot * 5 + failure_type]
                if failure_type == 0:
                    env.perturbations.perturbations[0].stuck_off_thruster(
                        failure_key,
                        envs_for_failure,
                        thrusters_for_failure,
                        start_time=start_time,
                    )
                elif failure_type == 1:
                    env.perturbations.perturbations[1].stuck_on_thruster(
                        failure_key,
                        envs_for_failure,
                        thrusters_for_failure,
                        start_time=start_time,
                    )
                else:
                    env.perturbations.perturbations[failure_type].register_perturbation(
                        failure_key,
                        envs_for_failure,
                        index=thruster,
                        start_time=start_time,
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
) -> int:
    num_perturbed = int(env.num_envs * max(0.0, min(1.0, fraction_perturbed_envs)))
    if num_perturbed <= 0:
        return 0

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
    )
    apply_failure_scenarios(
        env,
        key=apply_key,
        table=table,
        selected_envs=selected_envs,
        scenario_indices=scenario_indices,
        start_time=start_time,
    )
    return num_perturbed
