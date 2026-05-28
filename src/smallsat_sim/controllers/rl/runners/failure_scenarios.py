from __future__ import annotations

import csv
import hashlib
import os
import time
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

_STRICT_TIMING = bool(int(os.environ.get("SMALLSAT_STRICT_TIMING", "0")))

SPLIT_TRAIN = 0
SPLIT_SEMANTIC_EVAL = 1
SPLIT_STRESS_TEST = 2
BIN_EASY = 0
BIN_MEDIUM = 1
BIN_HARD = 2
BIN_NEAR_BOUNDARY = 3
REGIME_REDUNDANT = 0
REGIME_MARGINAL = 1
REGIME_AUTHORITY_LIMITED = 2
REGIME_BIAS_LIMITED = 3
LABEL_SYMMETRY_BREAKING = 1 << 0
LABEL_TORQUE_DEGENERATE = 1 << 1
LABEL_FORCE_DEGENERATE = 1 << 2
LABEL_COUPLED_FORCE_TORQUE = 1 << 3
LABEL_SATURATION_PRONE = 1 << 4
LABEL_NEAR_DEPENDENT = 1 << 5
LABEL_BIAS_DOMINATED = 1 << 6
LABEL_NONLINEAR_MISMATCH = 1 << 7
TASK_REGIME_EASY_FEASIBLE = 0
TASK_REGIME_HARD_FEASIBLE = 1
TASK_REGIME_NEAR_INFEASIBLE = 2
TASK_REGIME_INFEASIBLE = 3
FAILURE_TYPE_NAMES = {
    0: "stuck_off",
    1: "stuck_on",
    2: "faulty_valve",
    3: "saturated_thrust",
    4: "thrust_instability",
    5: "constant_disturbance",
}
SPLIT_NAMES = {
    SPLIT_TRAIN: "train",
    SPLIT_SEMANTIC_EVAL: "semantic_eval",
    SPLIT_STRESS_TEST: "stress_test",
}
BIN_NAMES = {
    BIN_EASY: "easy",
    BIN_MEDIUM: "medium",
    BIN_HARD: "hard",
    BIN_NEAR_BOUNDARY: "near_boundary",
}
AUTHORITY_REGIME_NAMES = {
    REGIME_REDUNDANT: "redundant",
    REGIME_MARGINAL: "marginal",
    REGIME_AUTHORITY_LIMITED: "authority_limited",
    REGIME_BIAS_LIMITED: "bias_limited",
}
TASK_REGIME_NAMES = {
    TASK_REGIME_EASY_FEASIBLE: "easy_feasible",
    TASK_REGIME_HARD_FEASIBLE: "hard_feasible",
    TASK_REGIME_NEAR_INFEASIBLE: "near_infeasible",
    TASK_REGIME_INFEASIBLE: "infeasible",
}
SCENARIO_FAILURE_STATUS = {
    0: PerturbationStatus.STUCK_OFF.value,
    1: PerturbationStatus.STUCK_ON.value,
    2: PerturbationStatus.FAULTY_VALVE.value,
    3: PerturbationStatus.SATURATED_THRUST.value,
    4: PerturbationStatus.THRUST_INSTABILITY.value,
}
_GP_SAMPLE_BANK: dict[tuple, tuple[jnp.ndarray, jnp.ndarray]] = {}
_SCENARIO_CACHE_VERSION = "task-conditioned-horizon-utilization-v6-semantic-splits"
TASK_ERROR_INFEASIBLE_THRESHOLD = 0.25
TASK_MARGIN_INFEASIBLE_THRESHOLD = -0.20
TASK_MARGIN_NEAR_INFEASIBLE_THRESHOLD = 0.0
TASK_MARGIN_HARD_FEASIBLE_THRESHOLD = 0.30
TASK_ERROR_HARD_FEASIBLE_THRESHOLD = 0.10
UTILIZATION_BIN_EASY_MAX = 0.30
UTILIZATION_BIN_MEDIUM_MAX = 0.60
UTILIZATION_BIN_HARD_MAX = 0.90


def _attach_scenario_apply_cache(
    table: dict[str, jnp.ndarray],
) -> dict[str, jnp.ndarray]:
    """Attach precomputed tensors used by per-epoch scenario application."""
    if "_apply_failure_valid_mask" in table and "_apply_failure_status" in table:
        return table

    num_rows = int(jnp.asarray(table["split"]).shape[0])
    if "failure_types" in table and "thrusters" in table:
        failure_types = jnp.asarray(table["failure_types"], dtype=jnp.int32)
        thrusters = jnp.asarray(table["thrusters"], dtype=jnp.int32)
    else:
        failure_types = jnp.full((num_rows, 1), -1, dtype=jnp.int32)
        thrusters = jnp.full((num_rows, 1), -1, dtype=jnp.int32)
    disturbance_wrench = jnp.asarray(
        table.get(
            "disturbance_wrench",
            jnp.zeros_like(table["targeted_task_wrench"]),
        ),
        dtype=jnp.float32,
    )

    failure_valid_mask = (failure_types >= 0) & (thrusters >= 0) & (failure_types != 5)
    # Lookup from failure_type -> perturbation status used in thruster_mask.
    failure_status_lut = jnp.asarray(
        [
            PerturbationStatus.STUCK_OFF.value,
            PerturbationStatus.STUCK_ON.value,
            PerturbationStatus.FAULTY_VALVE.value,
            PerturbationStatus.SATURATED_THRUST.value,
            PerturbationStatus.THRUST_INSTABILITY.value,
            -1,  # constant disturbance pseudo-failure is not a thruster perturbation
        ],
        dtype=jnp.int32,
    )
    clipped_types = jnp.clip(failure_types, 0, failure_status_lut.shape[0] - 1)
    failure_status = jnp.take(failure_status_lut, clipped_types)
    failure_status = jnp.where(failure_valid_mask, failure_status, -1)
    has_constant_disturbance = jnp.any((failure_types == 5) & (thrusters >= 0), axis=1)

    table["_apply_failure_valid_mask"] = failure_valid_mask
    table["_apply_failure_status"] = failure_status
    table["_apply_has_constant_disturbance"] = has_constant_disturbance
    table["_apply_disturbance_wrench"] = disturbance_wrench
    return table


def task_wrench_from_state_features(
    states: jnp.ndarray,
    *,
    kp_pos: float = 0.2,
    kd_pos: float = 1.0,
    kp_att: float = 3.0,
    kd_att: float = 5.0,
) -> jnp.ndarray:
    """
    PD-like regulation wrench used only for task-conditioned failure selection.

    ``states`` follows the environment convention:
    [p - p_ref, attitude_log_error, body_linear_velocity, body_angular_velocity].
    """
    states = jnp.asarray(states)
    force = -float(kp_pos) * states[..., 0:3] - float(kd_pos) * states[..., 6:9]
    torque = float(kp_att) * states[..., 3:6] - float(kd_att) * states[..., 9:12]
    return jnp.concatenate([force, torque], axis=-1)


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
    exhaustive_faults: int,
    sampled_per_fault_count: int,
    min_rank: int,
    stress_quantile: float,
    mild_effectiveness: float,
    include_infeasible: bool,
) -> str:
    digest = hashlib.sha256()
    digest.update(_SCENARIO_CACHE_VERSION.encode())
    digest.update(np.asarray(mixer_t, dtype=np.float32).tobytes())
    digest.update(np.asarray(ctrl_low, dtype=np.float32).tobytes())
    digest.update(np.asarray(ctrl_high, dtype=np.float32).tobytes())
    digest.update(str(int(max_faults)).encode())
    digest.update(str(int(exhaustive_faults)).encode())
    digest.update(str(int(sampled_per_fault_count)).encode())
    digest.update(str(int(min_rank)).encode())
    digest.update(f"{float(stress_quantile):.8f}".encode())
    digest.update(f"{float(mild_effectiveness):.8f}".encode())
    digest.update(str(int(bool(include_infeasible))).encode())
    return digest.hexdigest()[:16]


# Helper to generate a grid of unit directions in R^dim, including axes and random directions.
def _unit_direction_grid(dim: int = 6, random_count: int = 4096) -> np.ndarray:
    rng = np.random.default_rng(20260527)
    random_dirs = rng.normal(size=(int(random_count), int(dim))).astype(np.float32)
    random_dirs /= np.linalg.norm(random_dirs, axis=1, keepdims=True) + 1e-6

    axes: list[np.ndarray] = []
    for axis in range(dim):
        unit = np.zeros((dim,), dtype=np.float32)
        unit[axis] = 1.0
        axes.append(unit.copy())
        axes.append(-unit.copy())

    mixed: list[np.ndarray] = []
    for first in range(dim):
        for second in range(first + 1, dim):
            for sign_first, sign_second in (
                (1.0, 1.0),
                (1.0, -1.0),
                (-1.0, 1.0),
                (-1.0, -1.0),
            ):
                direction = np.zeros((dim,), dtype=np.float32)
                direction[first] = sign_first
                direction[second] = sign_second
                direction /= np.linalg.norm(direction) + 1e-6
                mixed.append(direction)

    return np.concatenate(
        [
            random_dirs,
            np.asarray(axes, dtype=np.float32),
            np.asarray(mixed, dtype=np.float32),
        ],
        axis=0,
    ).astype(np.float32)


def _task_wrench_samples(mixer_t: np.ndarray, ctrl_high: np.ndarray) -> np.ndarray:
    nominal_map = mixer_t.T
    positive = np.sum(np.maximum(nominal_map, 0.0) * ctrl_high[None, :], axis=1)
    negative = np.sum(np.minimum(nominal_map, 0.0) * ctrl_high[None, :], axis=1)
    axis_mag = 0.35 * np.minimum(np.abs(positive), np.abs(negative))
    axis_mag = np.maximum(axis_mag, 1e-3)

    direction_grid = _unit_direction_grid(dim=nominal_map.shape[0], random_count=4096)
    scale = float(np.median(axis_mag))
    task_wrenches = direction_grid * max(scale, 1e-3)

    canonical: list[np.ndarray] = []
    for axis in range(nominal_map.shape[0]):
        unit = np.zeros((nominal_map.shape[0],), dtype=np.float32)
        unit[axis] = axis_mag[axis]
        canonical.append(unit.copy())
        canonical.append(-unit.copy())

    return np.concatenate([task_wrenches, np.asarray(canonical, dtype=np.float32)], axis=0)


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
        for sign_first, sign_second in (
            (1.0, 1.0),
            (1.0, -1.0),
            (-1.0, 1.0),
            (-1.0, -1.0),
        ):
            wrench = np.zeros((nominal_map.shape[0],), dtype=np.float32)
            wrench[first] = sign_first * 0.5 * axis_mag[first]
            wrench[second] = sign_second * 0.5 * axis_mag[second]
            samples.append(wrench)

    random_dirs = _unit_direction_grid(dim=nominal_map.shape[0], random_count=1024)
    scale = float(np.median(axis_mag))
    samples.extend(list((random_dirs * max(scale, 1e-3)).astype(np.float32)))
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


def _projected_bounded_residuals(
    wrench_map: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_wrenches: np.ndarray,
    *,
    iterations: int = 32,
) -> np.ndarray:
    targets = np.asarray(target_wrenches, dtype=np.float32)
    pinv_t = np.linalg.pinv(wrench_map).T
    u = np.clip(targets @ pinv_t, lower[None, :], upper[None, :])
    lipschitz = float(np.linalg.norm(wrench_map, ord=2) ** 2) + 1e-6
    step_size = 1.0 / lipschitz
    for _ in range(iterations):
        residual = u @ wrench_map.T - targets
        grad = residual @ wrench_map
        u = np.clip(u - step_size * grad, lower[None, :], upper[None, :])
    return np.linalg.norm(u @ wrench_map.T - targets, axis=1).astype(np.float32)


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


def _max_feasible_wrench_scales(
    wrench_map: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_wrenches: np.ndarray,
    *,
    tolerance: float = 0.05,
    high: float = 4.0,
    iterations: int = 7,
) -> np.ndarray:
    targets = np.asarray(target_wrenches, dtype=np.float32)
    target_norms = np.linalg.norm(targets, axis=1) + 1e-6
    lo = np.zeros((targets.shape[0],), dtype=np.float32)
    hi = np.full((targets.shape[0],), float(high), dtype=np.float32)
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        scaled_targets = targets * mid[:, None]
        residuals = _projected_bounded_residuals(
            wrench_map,
            lower,
            upper,
            scaled_targets,
            iterations=16,
        )
        feasible = residuals / (mid * target_norms + 1e-6) <= tolerance
        lo = np.where(feasible, mid, lo)
        hi = np.where(feasible, hi, mid)
    return lo.astype(np.float32)


# Helper: minimum required utilization to achieve a target wrench along a horizon
def _min_required_utilization(
    wrench_map: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    target_wrench: np.ndarray,
    *,
    tolerance: float = 0.05,
    iterations: int = 10,
) -> tuple[float, float]:
    """Approximate min eta such that target_wrench is feasible with u <= eta * upper."""
    target_norm = float(np.linalg.norm(target_wrench)) + 1e-6
    if target_norm <= 1e-5:
        residual = _projected_bounded_residual(wrench_map, lower, upper, target_wrench)
        return 0.0, residual

    def feasible(eta: float) -> tuple[bool, float]:
        scaled_upper = np.minimum(upper, np.maximum(lower, eta * upper))
        residual = _projected_bounded_residual(
            wrench_map,
            lower,
            scaled_upper,
            target_wrench,
            iterations=24,
        )
        return residual / target_norm <= tolerance, residual

    ok_full, residual_full = feasible(1.0)
    if not ok_full:
        return float("inf"), residual_full

    lo = 0.0
    hi = 1.0
    residual_hi = residual_full
    for _ in range(iterations):
        mid = 0.5 * (lo + hi)
        ok_mid, residual_mid = feasible(mid)
        if ok_mid:
            hi = mid
            residual_hi = residual_mid
        else:
            lo = mid
    return float(hi), float(residual_hi)


def _scenario_horizon_wrenches(targeted_wrench: np.ndarray, horizon: int = 16) -> np.ndarray:
    horizon = max(1, int(horizon))
    scales = np.linspace(1.0, 0.20, horizon, dtype=np.float32)
    return scales[:, None] * targeted_wrench[None, :]


def _horizon_utilization_metrics(
    wrench_map: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    bias_wrench: np.ndarray,
    targeted_wrench: np.ndarray,
    *,
    horizon: int = 16,
) -> tuple[float, float, float]:
    etas = []
    residuals = []
    for wrench in _scenario_horizon_wrenches(targeted_wrench, horizon=horizon):
        net_wrench = wrench - bias_wrench
        eta, residual = _min_required_utilization(wrench_map, lower, upper, net_wrench)
        etas.append(eta)
        residuals.append(residual / (float(np.linalg.norm(net_wrench)) + 1e-6))
    etas_np = np.asarray(etas, dtype=np.float32)
    residuals_np = np.asarray(residuals, dtype=np.float32)
    return float(np.max(etas_np)), float(np.mean(etas_np)), float(np.max(residuals_np))


def _scenario_bounds(
    failure_types: tuple[int, ...],
    thrusters: tuple[int, ...],
    mixer_t: np.ndarray,
    ctrl_low: np.ndarray,
    ctrl_high: np.ndarray,
    mild_effectiveness: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    effectiveness = np.ones((mixer_t.shape[0],), dtype=np.float32)
    lower = ctrl_low.copy()
    upper = ctrl_high.copy()
    disturbance = np.zeros((mixer_t.shape[1],), dtype=np.float32)
    nominal_map = mixer_t.T
    for failure_type, thruster in zip(failure_types, thrusters, strict=True):
        if failure_type == 5:
            axis = int(thruster) % nominal_map.shape[0]
            sign = 1.0 if (int(thruster) // nominal_map.shape[0]) % 2 == 0 else -1.0
            positive = np.sum(np.maximum(nominal_map[axis], 0.0) * ctrl_high)
            negative = np.sum(np.minimum(nominal_map[axis], 0.0) * ctrl_high)
            axis_mag = 0.15 * min(abs(float(positive)), abs(float(negative)))
            disturbance[axis] += sign * max(axis_mag, 1e-3)
        elif failure_type == 0:
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
    return effectiveness, lower, upper, disturbance


def _authority_label_mask(
    failure_types: tuple[int, ...],
    min_sv: float,
    condition_number: float,
    bias_wrench_norm: float,
    targeted_utilization: float,
    targeted_direction: np.ndarray,
) -> int:
    label_mask = 0
    if len(failure_types) >= 2:
        label_mask |= LABEL_SYMMETRY_BREAKING
    force_norm = float(np.linalg.norm(targeted_direction[:3]))
    torque_norm = float(np.linalg.norm(targeted_direction[3:]))
    if torque_norm > 0.65:
        label_mask |= LABEL_TORQUE_DEGENERATE
    if force_norm > 0.65:
        label_mask |= LABEL_FORCE_DEGENERATE
    if force_norm > 0.25 and torque_norm > 0.25:
        label_mask |= LABEL_COUPLED_FORCE_TORQUE
    if targeted_utilization >= 0.75:
        label_mask |= LABEL_SATURATION_PRONE
    if condition_number > 50.0 or min_sv < 1e-3:
        label_mask |= LABEL_NEAR_DEPENDENT
    if bias_wrench_norm > 1e-4 or any(failure_type == 1 for failure_type in failure_types) or any(failure_type == 5 for failure_type in failure_types):
        label_mask |= LABEL_BIAS_DOMINATED
    if any(failure_type in (2, 3, 4) for failure_type in failure_types):
        label_mask |= LABEL_NONLINEAR_MISMATCH
    return int(label_mask)


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
    float,
    float,
    float,
    int,
    np.ndarray,
    np.ndarray,
]:
    effectiveness, lower, upper, disturbance = _scenario_bounds(
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

    task_net_wrenches = task_wrenches - disturbance[None, :]
    residuals = _projected_bounded_residuals(
        wrench_map,
        lower,
        upper,
        task_net_wrenches,
        iterations=32,
    )
    errors_np = residuals / (np.linalg.norm(task_wrenches, axis=1) + 1e-6)
    margins_np = (
        _max_feasible_wrench_scales(wrench_map, lower, upper, task_net_wrenches)
        - 1.0
    )

    targeted_net_wrenches = targeted_task_wrenches - disturbance[None, :]
    targeted_residuals = _projected_bounded_residuals(
        wrench_map,
        lower,
        upper,
        targeted_net_wrenches,
        iterations=32,
    )
    targeted_errors_np = targeted_residuals / (
        np.linalg.norm(targeted_task_wrenches, axis=1) + 1e-6
    )
    targeted_margins_np = (
        _max_feasible_wrench_scales(
            wrench_map,
            lower,
            upper,
            targeted_net_wrenches,
        )
        - 1.0
    )
    feasible_mask = targeted_errors_np <= 0.10
    hard_feasible_mask = np.logical_and(
        feasible_mask,
        np.logical_and(targeted_margins_np >= 0.0, targeted_margins_np <= 0.30),
    )
    if np.any(hard_feasible_mask):
        # Pick the feasible query closest to the damaged wrench boundary.
        scores = np.where(hard_feasible_mask, targeted_margins_np, np.inf)
        target_idx = int(np.argmin(scores))
    elif np.any(feasible_mask):
        # Fall back to the smallest positive-margin feasible query. This becomes
        # easy_feasible, not a fake hard case.
        scores = np.where(
            feasible_mask,
            np.maximum(targeted_margins_np, 0.0),
            np.inf,
        )
        target_idx = int(np.argmin(scores))
    else:
        # Only if no sampled query is feasible do we report an infeasible target.
        scores = targeted_errors_np + 0.05 * np.abs(targeted_margins_np)
        target_idx = int(np.argmin(scores))
    targeted_wrench = targeted_task_wrenches[target_idx].astype(np.float32)
    targeted_norm = float(np.linalg.norm(targeted_wrench))
    targeted_direction = targeted_wrench / max(targeted_norm, 1e-6)
    targeted_authority_scale = max(1.0 + float(targeted_margins_np[target_idx]), 1e-6)
    targeted_utilization = float(1.0 / targeted_authority_scale)
    bias_wrench = disturbance + wrench_map @ np.clip(np.zeros_like(lower), lower, upper)
    horizon_max_utilization, horizon_mean_utilization, horizon_max_error = _horizon_utilization_metrics(
        wrench_map,
        lower,
        upper,
        bias_wrench,
        targeted_wrench,
        horizon=16,
    )
    if np.isfinite(horizon_max_utilization):
        targeted_utilization = horizon_max_utilization
    zero_residual = _projected_bounded_residual(
        wrench_map, lower, upper, -disturbance
    )
    nominal_scale = float(np.median(np.linalg.norm(task_wrenches, axis=1))) + 1e-6
    bias_cancellation_error = zero_residual / nominal_scale
    # bias_wrench = disturbance + wrench_map @ np.clip(np.zeros_like(lower), lower, upper)
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
    authority_label_mask = _authority_label_mask(
        failure_types,
        min_sv,
        condition_number,
        bias_wrench_norm,
        targeted_utilization,
        targeted_direction,
    )
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
        targeted_utilization,
        horizon_max_utilization,
        horizon_mean_utilization,
        authority_label_mask,
        disturbance.astype(np.float32),
        targeted_direction.astype(np.float32),
    )


def _task_feasibility_regime(targeted_error: float, targeted_margin: float) -> int:
    if (
        targeted_error > TASK_ERROR_INFEASIBLE_THRESHOLD
        or targeted_margin < TASK_MARGIN_INFEASIBLE_THRESHOLD
    ):
        return TASK_REGIME_INFEASIBLE
    if targeted_margin < TASK_MARGIN_NEAR_INFEASIBLE_THRESHOLD:
        return TASK_REGIME_NEAR_INFEASIBLE
    if (
        targeted_margin <= TASK_MARGIN_HARD_FEASIBLE_THRESHOLD
        and targeted_error <= TASK_ERROR_HARD_FEASIBLE_THRESHOLD
    ):
        return TASK_REGIME_HARD_FEASIBLE
    return TASK_REGIME_EASY_FEASIBLE


def _failure_combo_key(
    combo: tuple[tuple[int, int], ...]
) -> tuple[tuple[int, int], ...]:
    return tuple(
        sorted(((int(ft), int(thr)) for ft, thr in combo), key=lambda item: item[1])
    )


def _iter_failure_combos(
    *,
    n_thrusters: int,
    max_faults: int,
    exhaustive_faults: int,
    sampled_per_fault_count: int,
) -> list[tuple[tuple[int, int], ...]]:
    max_faults = max(1, min(int(max_faults), int(n_thrusters)))
    exhaustive_faults = max(0, min(int(exhaustive_faults), max_faults))
    sampled_per_fault_count = max(0, int(sampled_per_fault_count))
    fault_pool = [
        (failure_type, thruster)
        for failure_type in range(5)
        for thruster in range(n_thrusters)
    ]
    # Constant disturbances are encoded as pseudo-thrusters 0..11:
    # axes 0..5 positive, axes 6..11 negative.
    fault_pool.extend((5, pseudo_axis) for pseudo_axis in range(12))
    combos_out: list[tuple[tuple[int, int], ...]] = []
    seen: set[tuple[tuple[int, int], ...]] = set()

    for n_faults in range(1, exhaustive_faults + 1):
        for combo in combinations(fault_pool, n_faults):
            real_thrusters = tuple(item[1] for item in combo if item[0] != 5)
            if len(set(real_thrusters)) != len(real_thrusters):
                continue
            key = _failure_combo_key(combo)
            if key in seen:
                continue
            seen.add(key)
            combos_out.append(key)

    if sampled_per_fault_count == 0:
        return combos_out

    rng = np.random.default_rng(20260526)
    for n_faults in range(exhaustive_faults + 1, max_faults + 1):
        accepted = 0
        attempts = 0
        max_attempts = sampled_per_fault_count * 50
        while accepted < sampled_per_fault_count and attempts < max_attempts:
            attempts += 1
            sampled_indices = rng.choice(len(fault_pool), size=n_faults, replace=False)
            raw_combo = tuple(fault_pool[int(idx)] for idx in sampled_indices)
            real_thrusters = tuple(item[1] for item in raw_combo if item[0] != 5)
            if len(set(real_thrusters)) != len(real_thrusters):
                continue
            combo = tuple(
                sorted(raw_combo, key=lambda item: (item[0] == 5, item[1], item[0]))
            )
            if combo in seen:
                continue
            seen.add(combo)
            combos_out.append(combo)
            accepted += 1

    return combos_out


@lru_cache(maxsize=16)
def _build_scenario_table_cached(
    mixer_t_bytes: bytes,
    mixer_t_shape: tuple[int, int],
    ctrl_low_bytes: bytes,
    ctrl_high_bytes: bytes,
    max_faults: int,
    exhaustive_faults: int,
    sampled_per_fault_count: int,
    min_rank: int,
    stress_quantile: float,
    mild_effectiveness: float,
    include_infeasible: bool,
) -> dict[str, np.ndarray]:
    mixer_t = np.frombuffer(mixer_t_bytes, dtype=np.float32).reshape(mixer_t_shape)
    ctrl_low = np.frombuffer(ctrl_low_bytes, dtype=np.float32).copy()
    ctrl_high = np.frombuffer(ctrl_high_bytes, dtype=np.float32).copy()
    task_wrenches = _task_wrench_samples(mixer_t, ctrl_high)
    targeted_task_wrenches = _targeted_task_wrench_samples(mixer_t, ctrl_high)
    n_thrusters = mixer_t.shape[0]
    table_width = int(max_faults)

    scenario_rows = []
    p90_all = []

    combos = _iter_failure_combos(
        n_thrusters=n_thrusters,
        max_faults=max_faults,
        exhaustive_faults=exhaustive_faults,
        sampled_per_fault_count=sampled_per_fault_count,
    )
    for combo in combos:
        n_faults = len(combo)
        thrusters = tuple(item[1] for item in combo)
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
            targeted_task_utilization,
            horizon_max_utilization,
            horizon_mean_utilization,
            authority_label_mask,
            disturbance_wrench,
            targeted_task_direction,
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
        task_feasibility_regime = _task_feasibility_regime(
            targeted_task_error,
            targeted_task_margin,
        )
        if task_feasibility_regime == TASK_REGIME_INFEASIBLE and not bool(
            include_infeasible
        ):
            continue
        scenario_rows.append(
            {
                "failure_types": failure_types,
                "thrusters": thrusters,
                "n_faults": n_faults,
                "active_thruster_count": n_thrusters
                - sum(1 for failure_type in failure_types if failure_type == 0),
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
                "targeted_task_utilization": targeted_task_utilization,
                "horizon_max_utilization": horizon_max_utilization,
                "horizon_mean_utilization": horizon_mean_utilization,
                "authority_label_mask": authority_label_mask,
                "disturbance_wrench": disturbance_wrench,
                "targeted_task_direction": targeted_task_direction,
                "task_feasibility_regime": task_feasibility_regime,
            }
        )
        p90_all.append(p90_error)

    if not scenario_rows:
        empty_int = np.empty((0, table_width), dtype=np.int32)
        return {
            "failure_types": empty_int,
            "thrusters": empty_int,
            "n_faults": np.empty((0,), dtype=np.int32),
            "active_thruster_count": np.empty((0,), dtype=np.int32),
            "split": np.empty((0,), dtype=np.int32),
            "is_stress": np.empty((0,), dtype=np.int32),
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
            "task_feasibility_regime": np.empty((0,), dtype=np.int32),
            "targeted_task_error": np.empty((0,), dtype=np.float32),
            "targeted_task_margin": np.empty((0,), dtype=np.float32),
            "targeted_task_wrench": np.empty((0, 6), dtype=np.float32),
            "targeted_task_utilization": np.empty((0,), dtype=np.float32),
            "horizon_max_utilization": np.empty((0,), dtype=np.float32),
            "horizon_mean_utilization": np.empty((0,), dtype=np.float32),
            "authority_label_mask": np.empty((0,), dtype=np.int32),
            "disturbance_wrench": np.empty((0, 6), dtype=np.float32),
            "targeted_task_direction": np.empty((0, 6), dtype=np.float32),
        }

    p90_all_np = np.asarray(p90_all, dtype=np.float32)
    stress_threshold = float(np.quantile(p90_all_np, stress_quantile))

    failure_type_rows: list[list[int]] = []
    thruster_rows: list[list[int]] = []
    n_fault_rows: list[int] = []
    active_thruster_count_rows: list[int] = []
    split_rows: list[int] = []
    is_stress_rows: list[int] = []
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
    task_feasibility_regime_rows: list[int] = []
    targeted_error_rows: list[float] = []
    targeted_margin_rows: list[float] = []
    targeted_wrench_rows: list[np.ndarray] = []
    targeted_utilization_rows: list[float] = []
    horizon_max_utilization_rows: list[float] = []
    horizon_mean_utilization_rows: list[float] = []
    authority_label_rows: list[int] = []
    disturbance_wrench_rows: list[np.ndarray] = []
    targeted_direction_rows: list[np.ndarray] = []

    for row_idx, row in enumerate(scenario_rows):
        failure_types = row["failure_types"]
        thrusters = row["thrusters"]
        n_faults = row["n_faults"]
        p90_error = row["p90_error"]
        del row_idx
        split = (
            SPLIT_STRESS_TEST
            if int(row["task_feasibility_regime"]) == TASK_REGIME_INFEASIBLE
            else SPLIT_TRAIN
        )
        is_stress = int(p90_error >= stress_threshold)

        utilization = float(row["targeted_task_utilization"])
        if utilization < UTILIZATION_BIN_EASY_MAX:
            difficulty_bin = BIN_EASY
        elif utilization < UTILIZATION_BIN_MEDIUM_MAX:
            difficulty_bin = BIN_MEDIUM
        elif utilization < UTILIZATION_BIN_HARD_MAX:
            difficulty_bin = BIN_HARD
        else:
            difficulty_bin = BIN_NEAR_BOUNDARY

        padded_types = list(failure_types) + [-1] * (table_width - n_faults)
        padded_thrusters = list(thrusters) + [-1] * (table_width - n_faults)
        failure_type_rows.append(padded_types)
        thruster_rows.append(padded_thrusters)
        n_fault_rows.append(n_faults)
        active_thruster_count_rows.append(row["active_thruster_count"])
        split_rows.append(split)
        is_stress_rows.append(is_stress)
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
        task_feasibility_regime_rows.append(row["task_feasibility_regime"])
        targeted_error_rows.append(row["targeted_task_error"])
        targeted_margin_rows.append(row["targeted_task_margin"])
        targeted_wrench_rows.append(row["targeted_task_wrench"])
        targeted_utilization_rows.append(row["targeted_task_utilization"])
        horizon_max_utilization_rows.append(row["horizon_max_utilization"])
        horizon_mean_utilization_rows.append(row["horizon_mean_utilization"])
        authority_label_rows.append(row["authority_label_mask"])
        disturbance_wrench_rows.append(row["disturbance_wrench"])
        targeted_direction_rows.append(row["targeted_task_direction"])

    return {
        "failure_types": np.asarray(failure_type_rows, dtype=np.int32),
        "thrusters": np.asarray(thruster_rows, dtype=np.int32),
        "n_faults": np.asarray(n_fault_rows, dtype=np.int32),
        "active_thruster_count": np.asarray(
            active_thruster_count_rows, dtype=np.int32
        ),
        "split": np.asarray(split_rows, dtype=np.int32),
        "is_stress": np.asarray(is_stress_rows, dtype=np.int32),
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
        "task_feasibility_regime": np.asarray(
            task_feasibility_regime_rows, dtype=np.int32
        ),
        "targeted_task_error": np.asarray(targeted_error_rows, dtype=np.float32),
        "targeted_task_margin": np.asarray(
            targeted_margin_rows, dtype=np.float32
        ),
        "targeted_task_wrench": np.asarray(
            targeted_wrench_rows, dtype=np.float32
        ),
        "targeted_task_utilization": np.asarray(
            targeted_utilization_rows, dtype=np.float32
        ),
        "horizon_max_utilization": np.asarray(
            horizon_max_utilization_rows, dtype=np.float32
        ),
        "horizon_mean_utilization": np.asarray(
            horizon_mean_utilization_rows, dtype=np.float32
        ),
        "authority_label_mask": np.asarray(authority_label_rows, dtype=np.int32),
        "disturbance_wrench": np.asarray(disturbance_wrench_rows, dtype=np.float32),
        "targeted_task_direction": np.asarray(
            targeted_direction_rows, dtype=np.float32
        ),
    }


def build_failure_scenario_table(
    thruster_mixer_t: jnp.ndarray,
    ctrl_low: jnp.ndarray,
    ctrl_high: jnp.ndarray,
    *,
    max_faults: int = 12,
    exhaustive_faults: int = 2,
    sampled_per_fault_count: int = 512,
    min_rank: int = 6,
    stress_quantile: float = 0.9,
    mild_effectiveness: float = 0.5,
    include_infeasible: bool = False,
) -> dict[str, jnp.ndarray]:
    mixer_t = np.asarray(jax.device_get(thruster_mixer_t), dtype=np.float32)
    ctrl_low_np = np.asarray(jax.device_get(ctrl_low), dtype=np.float32)
    ctrl_high_np = np.asarray(jax.device_get(ctrl_high), dtype=np.float32)
    max_faults = max(1, min(int(max_faults), mixer_t.shape[0]))
    exhaustive_faults = max(0, min(int(exhaustive_faults), max_faults))
    sampled_per_fault_count = max(0, int(sampled_per_fault_count))
    cache_key = _scenario_cache_key(
        mixer_t,
        ctrl_low_np,
        ctrl_high_np,
        max_faults=max_faults,
        exhaustive_faults=exhaustive_faults,
        sampled_per_fault_count=sampled_per_fault_count,
        min_rank=min_rank,
        stress_quantile=stress_quantile,
        mild_effectiveness=mild_effectiveness,
        include_infeasible=include_infeasible,
    )
    cache_path = os.path.join(_scenario_cache_dir(), f"{cache_key}.npz")
    if os.path.isfile(cache_path):
        print(f"[Failure Scenarios] Loading cached scenario table: {cache_path}", flush=True)
        with np.load(cache_path) as cached:
            return _attach_scenario_apply_cache(
                {key: jnp.asarray(cached[key]) for key in cached.files}
            )

    print(
        "[Failure Scenarios] Building scenario table "
        f"(max_faults={max_faults}, exhaustive_faults={exhaustive_faults}, "
        f"sampled_per_fault_count={sampled_per_fault_count}, min_rank={min_rank}); "
        "this is cached after the first run.",
        flush=True,
    )
    table = _build_scenario_table_cached(
        mixer_t.tobytes(),
        tuple(mixer_t.shape),
        ctrl_low_np.tobytes(),
        ctrl_high_np.tobytes(),
        max_faults,
        exhaustive_faults,
        sampled_per_fault_count,
        int(min_rank),
        float(stress_quantile),
        float(mild_effectiveness),
        bool(include_infeasible),
    )
    np.savez_compressed(cache_path, **table)
    print(f"[Failure Scenarios] Cached scenario table: {cache_path}", flush=True)
    return _attach_scenario_apply_cache(
        {key: jnp.asarray(value) for key, value in table.items()}
    )


def scenario_split_counts(table: dict[str, jnp.ndarray]) -> dict[str, int]:
    split = np.asarray(jax.device_get(table["split"]))
    stress = np.asarray(
        jax.device_get(table.get("is_stress", jnp.zeros_like(table["split"])))
    )
    return {
        "train": int((split == SPLIT_TRAIN).sum()),
        "semantic_eval": int((split == SPLIT_SEMANTIC_EVAL).sum()),
        "stress_test": int((split == SPLIT_STRESS_TEST).sum()),
        "stress": int(stress.sum()),
        "semantic_eval_stress": int(
            np.logical_and(split == SPLIT_SEMANTIC_EVAL, stress.astype(bool)).sum()
        ),
        "stress_test_stress": int(
            np.logical_and(split == SPLIT_STRESS_TEST, stress.astype(bool)).sum()
        ),
    }


def scenario_bin_counts(table: dict[str, jnp.ndarray], split_id: int = SPLIT_TRAIN) -> dict[str, int]:
    split = np.asarray(jax.device_get(table["split"]))
    bins = np.asarray(jax.device_get(table["difficulty_bin"]))
    split_mask = split == split_id
    return {
        "easy": int(np.logical_and(split_mask, bins == BIN_EASY).sum()),
        "medium": int(np.logical_and(split_mask, bins == BIN_MEDIUM).sum()),
        "hard": int(np.logical_and(split_mask, bins == BIN_HARD).sum()),
        "near_boundary": int(
            np.logical_and(split_mask, bins == BIN_NEAR_BOUNDARY).sum()
        ),
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


def scenario_task_regime_counts(
    table: dict[str, jnp.ndarray],
    split_id: int | None = None,
) -> dict[str, int]:
    regimes = np.asarray(jax.device_get(table["task_feasibility_regime"]))
    if split_id is None:
        mask = np.ones_like(regimes, dtype=bool)
    else:
        split = np.asarray(jax.device_get(table["split"]))
        mask = split == int(split_id)
    return {
        name: int(np.logical_and(mask, regimes == regime_id).sum())
        for regime_id, name in TASK_REGIME_NAMES.items()
    }


def save_scenario_table_csv(table: dict[str, jnp.ndarray], path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    failure_types = np.asarray(jax.device_get(table["failure_types"]))
    thrusters = np.asarray(jax.device_get(table["thrusters"]))
    n_faults = np.asarray(jax.device_get(table["n_faults"]))
    active_thruster_count = np.asarray(jax.device_get(table["active_thruster_count"]))
    splits = np.asarray(jax.device_get(table["split"]))
    is_stress = np.asarray(
        jax.device_get(table.get("is_stress", jnp.zeros_like(table["split"])))
    )
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
    task_feasibility_regime = np.asarray(
        jax.device_get(table["task_feasibility_regime"])
    )
    targeted_error = np.asarray(jax.device_get(table["targeted_task_error"]))
    targeted_margin = np.asarray(jax.device_get(table["targeted_task_margin"]))
    targeted_wrench = np.asarray(jax.device_get(table["targeted_task_wrench"]))
    targeted_utilization = np.asarray(
        jax.device_get(
            table.get("targeted_task_utilization", jnp.zeros_like(targeted_error))
        )
    )
    horizon_max_utilization = np.asarray(
        jax.device_get(table.get("horizon_max_utilization", targeted_utilization))
    )
    horizon_mean_utilization = np.asarray(
        jax.device_get(table.get("horizon_mean_utilization", targeted_utilization))
    )
    targeted_direction = np.asarray(
        jax.device_get(
            table.get("targeted_task_direction", jnp.zeros_like(targeted_wrench))
        )
    )
    authority_label_mask = np.asarray(
        jax.device_get(
            table.get("authority_label_mask", jnp.zeros_like(table["difficulty_bin"]))
        )
    )
    disturbance_wrench = np.asarray(
        jax.device_get(
            table.get("disturbance_wrench", jnp.zeros_like(targeted_wrench))
        )
    )

    max_faults = failure_types.shape[1] if failure_types.ndim == 2 else 0
    fieldnames = [
        "scenario_id",
        "split",
        "difficulty_bin",
        "n_faults",
        "active_thruster_count",
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
        "task_feasibility_regime",
        "targeted_task_error",
        "targeted_task_margin",
        "targeted_task_utilization",
        "horizon_max_utilization",
        "horizon_mean_utilization",
        "authority_label_mask",
        "is_stress",
    ]
    fieldnames.extend([f"targeted_wrench_{idx}" for idx in range(6)])
    fieldnames.extend([f"targeted_direction_{idx}" for idx in range(6)])
    fieldnames.extend([f"disturbance_wrench_{idx}" for idx in range(6)])
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
                "active_thruster_count": int(active_thruster_count[scenario_id]),
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
                "task_feasibility_regime": TASK_REGIME_NAMES.get(
                    int(task_feasibility_regime[scenario_id]), "unknown"
                ),
                "targeted_task_error": float(targeted_error[scenario_id]),
                "targeted_task_margin": float(targeted_margin[scenario_id]),
                "targeted_task_utilization": float(
                    targeted_utilization[scenario_id]
                ),
                "horizon_max_utilization": float(
                    horizon_max_utilization[scenario_id]
                ),
                "horizon_mean_utilization": float(
                    horizon_mean_utilization[scenario_id]
                ),
                "authority_label_mask": int(authority_label_mask[scenario_id]),
                "is_stress": bool(is_stress[scenario_id]),
            }
            for wrench_idx in range(6):
                row[f"targeted_wrench_{wrench_idx}"] = float(
                    targeted_wrench[scenario_id, wrench_idx]
                )
                row[f"targeted_direction_{wrench_idx}"] = float(
                    targeted_direction[scenario_id, wrench_idx]
                )
                row[f"disturbance_wrench_{wrench_idx}"] = float(
                    disturbance_wrench[scenario_id, wrench_idx]
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
    scenario_indices = jnp.asarray(scenario_indices, dtype=jnp.int32)
    scenario_indices_np = np.asarray(jax.device_get(scenario_indices), dtype=np.int32)
    if scenario_indices_np.size == 0:
        return {
            f"{prefix}/num_selected": 0.0,
            f"{prefix}/bin_easy_fraction": 0.0,
            f"{prefix}/bin_medium_fraction": 0.0,
            f"{prefix}/bin_hard_fraction": 0.0,
            f"{prefix}/bin_near_boundary_fraction": 0.0,
        }

    def selected_np(values: jnp.ndarray) -> np.ndarray:
        return np.asarray(jax.device_get(jnp.asarray(values)[scenario_indices]))

    bins = selected_np(table["difficulty_bin"])
    splits = selected_np(table["split"])
    is_stress = np.asarray(
        selected_np(table.get("is_stress", jnp.zeros_like(table["split"])))
    )
    n_faults = selected_np(table["n_faults"])
    failure_types = selected_np(table["failure_types"])
    active_thruster_count = selected_np(table["active_thruster_count"])
    p90_error = selected_np(table["p90_feasibility_error"])
    mean_error = selected_np(table["mean_feasibility_error"])
    bias_error = selected_np(table["bias_cancellation_error"])
    bias_norm = selected_np(table["bias_wrench_norm"])
    p10_margin = selected_np(table["p10_authority_margin"])
    mean_margin = selected_np(table["mean_authority_margin"])
    targeted_error = selected_np(table["targeted_task_error"])
    targeted_margin = selected_np(table["targeted_task_margin"])
    targeted_utilization = selected_np(
        table.get(
            "targeted_task_utilization",
            jnp.zeros_like(table["targeted_task_margin"]),
        )
    )
    horizon_max_utilization = selected_np(
        table.get("horizon_max_utilization", table["targeted_task_utilization"])
    )
    horizon_mean_utilization = selected_np(
        table.get("horizon_mean_utilization", table["targeted_task_utilization"])
    )
    authority_label_mask = selected_np(
        table.get("authority_label_mask", jnp.zeros_like(table["difficulty_bin"]))
    )
    regimes = selected_np(table["authority_regime"])
    task_regimes = selected_np(table["task_feasibility_regime"])
    denom = float(max(scenario_indices_np.size, 1))
    return {
        f"{prefix}/num_selected": float(scenario_indices_np.size),
        f"{prefix}/bin_easy_fraction": float((bins == BIN_EASY).sum() / denom),
        f"{prefix}/bin_medium_fraction": float((bins == BIN_MEDIUM).sum() / denom),
        f"{prefix}/bin_hard_fraction": float((bins == BIN_HARD).sum() / denom),
        f"{prefix}/bin_near_boundary_fraction": float(
            (bins == BIN_NEAR_BOUNDARY).sum() / denom
        ),
        f"{prefix}/split_train_fraction": float((splits == SPLIT_TRAIN).sum() / denom),
        f"{prefix}/split_semantic_eval_fraction": float(
            (splits == SPLIT_SEMANTIC_EVAL).sum() / denom
        ),
        f"{prefix}/split_stress_test_fraction": float(
            (splits == SPLIT_STRESS_TEST).sum() / denom
        ),
        f"{prefix}/stress_fraction": float(is_stress.sum() / denom),
        f"{prefix}/mean_num_faults": float(n_faults.mean()),
        f"{prefix}/mean_active_thruster_count": float(active_thruster_count.mean()),
        **{
            f"{prefix}/failure_type_{name}_fraction": float(
                np.any(failure_types == failure_type, axis=1).sum() / denom
            )
            for failure_type, name in FAILURE_TYPE_NAMES.items()
        },
        f"{prefix}/mean_p90_feasibility_error": float(p90_error.mean()),
        f"{prefix}/max_p90_feasibility_error": float(p90_error.max()),
        f"{prefix}/mean_feasibility_error": float(mean_error.mean()),
        f"{prefix}/mean_bias_cancellation_error": float(bias_error.mean()),
        f"{prefix}/mean_bias_wrench_norm": float(bias_norm.mean()),
        f"{prefix}/mean_p10_authority_margin": float(p10_margin.mean()),
        f"{prefix}/mean_authority_margin": float(mean_margin.mean()),
        f"{prefix}/mean_targeted_task_error": float(targeted_error.mean()),
        f"{prefix}/mean_targeted_task_margin": float(targeted_margin.mean()),
        f"{prefix}/mean_targeted_task_utilization": float(
            targeted_utilization.mean()
        ),
        f"{prefix}/mean_horizon_max_utilization": float(
            horizon_max_utilization.mean()
        ),
        f"{prefix}/mean_horizon_mean_utilization": float(
            horizon_mean_utilization.mean()
        ),
        f"{prefix}/label_symmetry_breaking_fraction": float(
            ((authority_label_mask & LABEL_SYMMETRY_BREAKING) != 0).sum() / denom
        ),
        f"{prefix}/label_torque_degenerate_fraction": float(
            ((authority_label_mask & LABEL_TORQUE_DEGENERATE) != 0).sum() / denom
        ),
        f"{prefix}/label_force_degenerate_fraction": float(
            ((authority_label_mask & LABEL_FORCE_DEGENERATE) != 0).sum() / denom
        ),
        f"{prefix}/label_coupled_force_torque_fraction": float(
            ((authority_label_mask & LABEL_COUPLED_FORCE_TORQUE) != 0).sum() / denom
        ),
        f"{prefix}/label_saturation_prone_fraction": float(
            ((authority_label_mask & LABEL_SATURATION_PRONE) != 0).sum() / denom
        ),
        f"{prefix}/label_near_dependent_fraction": float(
            ((authority_label_mask & LABEL_NEAR_DEPENDENT) != 0).sum() / denom
        ),
        f"{prefix}/label_bias_dominated_fraction": float(
            ((authority_label_mask & LABEL_BIAS_DOMINATED) != 0).sum() / denom
        ),
        f"{prefix}/label_nonlinear_mismatch_fraction": float(
            ((authority_label_mask & LABEL_NONLINEAR_MISMATCH) != 0).sum() / denom
        ),
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
        f"{prefix}/task_regime_easy_feasible_fraction": float(
            (task_regimes == TASK_REGIME_EASY_FEASIBLE).sum() / denom
        ),
        f"{prefix}/task_regime_hard_feasible_fraction": float(
            (task_regimes == TASK_REGIME_HARD_FEASIBLE).sum() / denom
        ),
        f"{prefix}/task_regime_near_infeasible_fraction": float(
            (task_regimes == TASK_REGIME_NEAR_INFEASIBLE).sum() / denom
        ),
        f"{prefix}/task_regime_infeasible_fraction": float(
            (task_regimes == TASK_REGIME_INFEASIBLE).sum() / denom
        ),
    }


def targeted_pose_errors_from_scenarios(
    table: dict[str, jnp.ndarray],
    scenario_indices: jnp.ndarray,
    *,
    distance: float,
    attitude_scale: float = 0.8,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
    Convert each scenario's certified task wrench into a matching initial error.

    The selected scenario task wrench is the wrench for which feasibility was
    checked offline. Starting opposite the force direction makes the regulation
    task demand that force direction. Starting with attitude error aligned to
    the torque direction makes the attitude task demand that torque direction.
    """
    wrenches = table["targeted_task_wrench"][scenario_indices]
    translation = wrenches[:, :3]
    torque = wrenches[:, 3:6]
    translation_norm = jnp.linalg.norm(translation, axis=1, keepdims=True)
    torque_norm = jnp.linalg.norm(torque, axis=1, keepdims=True)
    translation_fallback = jnp.tile(
        jnp.array([[1.0, 0.0, 0.0]], dtype=jnp.float32),
        (translation.shape[0], 1),
    )
    torque_fallback = jnp.tile(
        jnp.array([[0.0, 0.0, 1.0]], dtype=jnp.float32),
        (torque.shape[0], 1),
    )
    translation_direction = jnp.where(
        translation_norm > 1e-6,
        translation / (translation_norm + 1e-6),
        translation_fallback,
    )
    torque_direction = jnp.where(
        torque_norm > 1e-6,
        torque / (torque_norm + 1e-6),
        torque_fallback,
    )
    position_offset = -float(distance) * translation_direction
    attitude_error = float(attitude_scale) * torque_direction
    return position_offset, attitude_error


def sample_scenario_indices(
    key: jnp.ndarray,
    table: dict[str, jnp.ndarray],
    *,
    split_id: int,
    count: int,
    difficulty_bin: int | None = None,
    authority_regime: int | None = None,
    task_feasibility_regime: int | None = None,
    failure_type: int | None = None,
    authority_label_any_mask: int | None = None,
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
    if task_feasibility_regime is not None:
        task_mask = jnp.logical_and(
            mask, table["task_feasibility_regime"] == int(task_feasibility_regime)
        )
        mask = jnp.where(jnp.any(task_mask), task_mask, mask)
    if failure_type is not None:
        type_mask = jnp.logical_and(
            mask, jnp.any(table["failure_types"] == int(failure_type), axis=1)
        )
        mask = jnp.where(jnp.any(type_mask), type_mask, mask)
    if authority_label_any_mask is not None:
        label_mask = jnp.logical_and(
            mask,
            (table["authority_label_mask"] & int(authority_label_any_mask)) != 0,
        )
        mask = jnp.where(jnp.any(label_mask), label_mask, mask)
    logits = jnp.where(mask, 0.0, -jnp.inf)
    sampled = jax.random.categorical(key, logits, shape=(count,))
    return sampled.astype(jnp.int32)


def sample_task_conditioned_scenario_indices(
    key: jnp.ndarray,
    table: dict[str, jnp.ndarray],
    *,
    split_id: int,
    task_wrenches: jnp.ndarray,
    difficulty_bin: int | None = None,
    authority_regime: int | None = None,
    task_feasibility_regime: int | None = None,
    failure_type: int | None = None,
    authority_label_any_mask: int | None = None,
    alignment_temperature: float = 8.0,
) -> jnp.ndarray:
    """
    Sample one scenario per task wrench, favoring failures weak along that task.

    This is the online part of the library approach: the table is precomputed,
    but reset-time selection is conditioned on the actual regulation demand.
    """
    task_wrenches = jnp.asarray(task_wrenches, dtype=jnp.float32)
    count = int(task_wrenches.shape[0])
    if count == 0:
        return jnp.empty((0,), dtype=jnp.int32)

    mask = table["split"] == int(split_id)
    if difficulty_bin is not None:
        bin_mask = jnp.logical_and(mask, table["difficulty_bin"] == int(difficulty_bin))
        mask = jnp.where(jnp.any(bin_mask), bin_mask, mask)
    if authority_regime is not None:
        regime_mask = jnp.logical_and(
            mask, table["authority_regime"] == int(authority_regime)
        )
        mask = jnp.where(jnp.any(regime_mask), regime_mask, mask)
    if task_feasibility_regime is not None:
        task_mask = jnp.logical_and(
            mask, table["task_feasibility_regime"] == int(task_feasibility_regime)
        )
        mask = jnp.where(jnp.any(task_mask), task_mask, mask)
    if failure_type is not None:
        type_mask = jnp.logical_and(
            mask, jnp.any(table["failure_types"] == int(failure_type), axis=1)
        )
        mask = jnp.where(jnp.any(type_mask), type_mask, mask)
    if authority_label_any_mask is not None:
        label_mask = jnp.logical_and(
            mask,
            (table["authority_label_mask"] & int(authority_label_any_mask)) != 0,
        )
        mask = jnp.where(jnp.any(label_mask), label_mask, mask)

    feasible_mask = jnp.logical_and(
        mask, table["task_feasibility_regime"] != TASK_REGIME_INFEASIBLE
    )
    mask = jnp.where(jnp.any(feasible_mask), feasible_mask, mask)

    scenario_dirs = table.get("targeted_task_direction")
    if scenario_dirs is None:
        scenario_dirs = table["targeted_task_wrench"] / (
            jnp.linalg.norm(table["targeted_task_wrench"], axis=1, keepdims=True)
            + 1e-6
        )
    task_dirs = task_wrenches / (
        jnp.linalg.norm(task_wrenches, axis=1, keepdims=True) + 1e-6
    )
    alignment = task_dirs @ scenario_dirs.T
    base_logits = jnp.where(mask, 0.0, -jnp.inf)
    logits = base_logits[None, :] + float(alignment_temperature) * alignment
    return jax.random.categorical(key, logits, axis=1).astype(jnp.int32)


def _scenario_rows_for_indices(
    table: dict[str, jnp.ndarray],
    selected_envs: jnp.ndarray,
    scenario_indices: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    table = _attach_scenario_apply_cache(table)
    selected_envs = jnp.asarray(selected_envs, dtype=jnp.int32)
    scenario_indices = jnp.asarray(scenario_indices, dtype=jnp.int32)
    thrusters = jnp.asarray(table["thrusters"], dtype=jnp.int32)[scenario_indices]
    valid = jnp.asarray(table["_apply_failure_valid_mask"], dtype=bool)[scenario_indices]
    status = jnp.asarray(table["_apply_failure_status"], dtype=jnp.int32)[scenario_indices]

    env_rows = jnp.broadcast_to(selected_envs[:, None], thrusters.shape)
    flat_envs = env_rows[valid].astype(jnp.int32)
    flat_thrusters = thrusters[valid].astype(jnp.int32)
    flat_status = status[valid].astype(jnp.int32)
    return flat_envs, flat_thrusters, flat_status


# Helper for constant disturbance pseudo-failures (failure_type == 5)
def _constant_disturbance_rows_for_indices(
    table: dict[str, jnp.ndarray],
    selected_envs: jnp.ndarray,
    scenario_indices: jnp.ndarray,
    *,
    wrench_dim: int = 6,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return per-env additive wrench disturbances encoded in selected scenarios.

    Offline constant disturbances are encoded as failure_type == 5 with pseudo
    thrusters 0..11: axes 0..5 are positive, axes 6..11 are negative. This
    helper converts those pseudo-failures into one 6D disturbance wrench per
    selected environment. The actual magnitude matches `_scenario_bounds`.
    """
    table = _attach_scenario_apply_cache(table)
    selected_envs = jnp.asarray(selected_envs, dtype=jnp.int32)
    scenario_indices = jnp.asarray(scenario_indices, dtype=jnp.int32)
    if selected_envs.size == 0:
        return selected_envs, jnp.zeros((0, wrench_dim), dtype=jnp.float32)

    disturbance_wrenches = jnp.asarray(
        table["_apply_disturbance_wrench"], dtype=jnp.float32
    )[scenario_indices]
    has_disturbance = jnp.asarray(
        table["_apply_has_constant_disturbance"], dtype=bool
    )[scenario_indices][:, None]
    disturbances = jnp.where(
        has_disturbance,
        disturbance_wrenches[:, :wrench_dim],
        jnp.zeros((selected_envs.shape[0], wrench_dim), dtype=jnp.float32),
    )

    return selected_envs, disturbances


def _apply_constant_disturbance_scenarios(
    env,
    table: dict[str, jnp.ndarray],
    selected_envs: jnp.ndarray,
    scenario_indices: jnp.ndarray,
) -> None:
    """Apply constant-disturbance pseudo-failures if the environment supports them.

    This intentionally uses optional hooks/attributes so the failure-library code
    does not hard-depend on a single disturbance implementation. If no supported
    runtime disturbance interface exists, the scenario table still works for
    thruster failures and the constant-disturbance pseudo-failures are ignored at
    application time.
    """
    env_rows, disturbances = _constant_disturbance_rows_for_indices(
        table, selected_envs, scenario_indices
    )
    if int(env_rows.shape[0]) == 0 or not bool(jnp.any(jnp.abs(disturbances) > 0.0)):
        return

    if hasattr(env, "set_constant_wrench_disturbances"):
        env.set_constant_wrench_disturbances(env_rows, disturbances)
        return

    if hasattr(env, "set_constant_disturbances"):
        env.set_constant_disturbances(env_rows, disturbances)
        return

    if hasattr(env, "constant_wrench_disturbance"):
        current = jnp.asarray(env.constant_wrench_disturbance)
        if current.ndim == 2 and current.shape[-1] >= disturbances.shape[-1]:
            current = current.at[env_rows, : disturbances.shape[-1]].set(disturbances)
            env.constant_wrench_disturbance = current
            return

    if hasattr(env, "constant_disturbance"):
        current = jnp.asarray(env.constant_disturbance)
        if current.ndim == 2 and current.shape[-1] >= disturbances.shape[-1]:
            current = current.at[env_rows, : disturbances.shape[-1]].set(disturbances)
            env.constant_disturbance = current
            return

    if hasattr(env, "disturbance_wrench"):
        current = jnp.asarray(env.disturbance_wrench)
        if current.ndim == 2 and current.shape[-1] >= disturbances.shape[-1]:
            current = current.at[env_rows, : disturbances.shape[-1]].set(disturbances)
            env.disturbance_wrench = current
            return


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
) -> dict[str, float]:
    timing = {
        "disturbance": 0.0,
        "expand_scatter": 0.0,
        "state_build": 0.0,
        "refresh": 0.0,
    }
    if selected_envs.size == 0:
        return timing

    timing_start = time.perf_counter()
    _apply_constant_disturbance_scenarios(
        env,
        table,
        selected_envs,
        scenario_indices,
    )
    timing["disturbance"] = time.perf_counter() - timing_start

    timing_start = time.perf_counter()
    table = _attach_scenario_apply_cache(table)
    selected_envs = jnp.asarray(selected_envs, dtype=jnp.int32)
    scenario_indices = jnp.asarray(scenario_indices, dtype=jnp.int32)
    thrusters = jnp.asarray(table["thrusters"], dtype=jnp.int32)[scenario_indices]
    valid = jnp.asarray(table["_apply_failure_valid_mask"], dtype=bool)[
        scenario_indices
    ]
    status = jnp.asarray(table["_apply_failure_status"], dtype=jnp.int32)[
        scenario_indices
    ]
    if not bool(jnp.any(valid)):
        if hasattr(env, "_refresh_effect_states"):
            env._refresh_effect_states()
            env._state = env._state.replace(perturbation_states=env.perturbation_states)
        timing["refresh"] = time.perf_counter() - timing_start
        return timing
    env_rows = jnp.broadcast_to(selected_envs[:, None], thrusters.shape)
    safe_envs = jnp.where(valid, env_rows, 0).astype(jnp.int32)
    safe_thrusters = jnp.where(valid, thrusters, 0).astype(jnp.int32)

    # Update the shared env/thruster failure mask once. This avoids the slow
    # interactive registration path, which grouped scenarios by type/thruster and
    # repeatedly synchronized with the host.
    thruster_mask = jnp.asarray(Perturbation.thruster_mask)
    status_updates = jnp.zeros_like(thruster_mask).at[
        safe_envs, safe_thrusters
    ].max(jnp.where(valid, status, 0))
    Perturbation.thruster_mask = jnp.where(
        status_updates != 0, status_updates, thruster_mask
    )
    if _STRICT_TIMING:
        jax.block_until_ready(Perturbation.thruster_mask)
    timing["expand_scatter"] = time.perf_counter() - timing_start

    timing_start = time.perf_counter()
    start_time_value = jnp.asarray(
        0.0 if start_time is None else start_time, dtype=jnp.float32
    )
    subkeys = jax.random.split(key, 6)
    perturbations = env.perturbations.perturbations

    for failure_type, failure_status in SCENARIO_FAILURE_STATUS.items():
        type_mask = jnp.logical_and(valid, status == int(failure_status))
        perturbation = perturbations[failure_type]
        base_start_times = getattr(
            perturbation,
            "start_times",
            jnp.zeros((env.num_envs, env.act_dim), dtype=jnp.float32),
        )
        start_times = jnp.zeros_like(base_start_times).at[
            safe_envs, safe_thrusters
        ].max(jnp.where(type_mask, start_time_value.astype(base_start_times.dtype), 0.0))

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
            min_force = perturbation.min_thruster_force[safe_thrusters]
            max_force = perturbation.max_thruster_force[safe_thrusters]
            sampled_force = jax.random.uniform(
                subkeys[1],
                shape=safe_thrusters.shape,
                minval=min_force,
                maxval=max_force,
            )
            force_mask = jnp.zeros(stuck_force.shape, dtype=bool).at[
                safe_envs, safe_thrusters
            ].max(type_mask)
            force_updates = jnp.zeros_like(stuck_force).at[
                safe_envs, safe_thrusters
            ].max(jnp.where(type_mask, sampled_force, 0.0))
            stuck_force = jnp.where(force_mask, force_updates, stuck_force)
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
    timing["state_build"] = time.perf_counter() - timing_start

    timing_start = time.perf_counter()
    env._refresh_effect_states()
    env._state = env._state.replace(perturbation_states=env.perturbation_states)
    timing["refresh"] = time.perf_counter() - timing_start
    return timing


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
    task_feasibility_regime: int | None = None,
    failure_type: int | None = None,
    authority_label_any_mask: int | None = None,
    failure_sampling_mix: tuple[dict, ...] | list[dict] | None = None,
    task_wrenches: jnp.ndarray | None = None,
    return_selection: bool = False,
) -> dict[str, float] | tuple[dict[str, float], jnp.ndarray, jnp.ndarray]:
    select_duration = 0.0
    sample_duration = 0.0
    apply_duration = 0.0
    payload_duration = 0.0
    apply_detail = {
        "disturbance": 0.0,
        "expand_scatter": 0.0,
        "state_build": 0.0,
        "refresh": 0.0,
    }
    num_perturbed = int(env.num_envs * max(0.0, min(1.0, fraction_perturbed_envs)))
    if num_perturbed <= 0:
        payload = scenario_selection_payload(
            table,
            jnp.empty((0,), dtype=jnp.int32),
        )
        payload["_timing/select"] = select_duration
        payload["_timing/sample"] = sample_duration
        payload["_timing/apply"] = apply_duration
        payload["_timing/payload"] = payload_duration
        payload["_timing/apply_disturbance"] = apply_detail["disturbance"]
        payload["_timing/apply_expand_scatter"] = apply_detail["expand_scatter"]
        payload["_timing/apply_state_build"] = apply_detail["state_build"]
        payload["_timing/apply_refresh"] = apply_detail["refresh"]
        if return_selection:
            empty = jnp.empty((0,), dtype=jnp.int32)
            return payload, empty, empty
        return payload

    select_key, scenario_key, apply_key = jax.random.split(key, 3)
    timing_start = time.perf_counter()
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
    if _STRICT_TIMING:
        jax.block_until_ready(selected_envs)
    select_duration = time.perf_counter() - timing_start
    timing_start = time.perf_counter()
    if failure_sampling_mix:
        mix_entries = [
            entry for entry in failure_sampling_mix if int(entry["weight"]) > 0
        ]
        if not mix_entries:
            mix_entries = [
                {"weight": 1, "task_feasibility_regime": task_feasibility_regime}
            ]
        weights_np = np.asarray(
            [float(entry["weight"]) for entry in mix_entries], dtype=np.float64
        )
        weights_np = weights_np / max(float(weights_np.sum()), 1e-8)
        raw_counts = weights_np * float(num_perturbed)
        mix_counts = np.floor(raw_counts).astype(np.int32)
        remainder = int(num_perturbed - int(mix_counts.sum()))
        if remainder > 0:
            order = np.argsort(-(raw_counts - mix_counts))
            mix_counts[order[:remainder]] += 1
        mix_choice_np = np.concatenate(
            [
                np.full(int(count), mix_idx, dtype=np.int32)
                for mix_idx, count in enumerate(mix_counts)
                if int(count) > 0
            ]
        )
        mix_choice = jax.random.permutation(
            scenario_key, jnp.asarray(mix_choice_np, dtype=jnp.int32)
        )
        scenario_indices = jnp.zeros((num_perturbed,), dtype=jnp.int32)
        scenario_keys = jax.random.split(scenario_key, max(len(mix_entries), 1))
        selected_task_wrenches = (
            None if task_wrenches is None else jnp.asarray(task_wrenches)[selected_envs]
        )
        for mix_idx, mix_entry in enumerate(mix_entries):
            env_mask = mix_choice == mix_idx
            env_count = int(jnp.sum(env_mask))
            if env_count <= 0:
                continue
            sampled = (
                sample_scenario_indices(
                    scenario_keys[mix_idx],
                    table,
                    split_id=split_id,
                    count=env_count,
                    difficulty_bin=difficulty_bin,
                    authority_regime=authority_regime,
                    task_feasibility_regime=mix_entry.get("task_feasibility_regime"),
                    failure_type=failure_type,
                    authority_label_any_mask=mix_entry.get("authority_label_any_mask"),
                )
                if selected_task_wrenches is None
                else sample_task_conditioned_scenario_indices(
                    scenario_keys[mix_idx],
                    table,
                    split_id=split_id,
                    task_wrenches=selected_task_wrenches[env_mask],
                    difficulty_bin=difficulty_bin,
                    authority_regime=authority_regime,
                    task_feasibility_regime=mix_entry.get("task_feasibility_regime"),
                    failure_type=failure_type,
                    authority_label_any_mask=mix_entry.get("authority_label_any_mask"),
                )
            )
            scenario_indices = scenario_indices.at[jnp.where(env_mask, size=env_count)[0]].set(
                sampled
            )
    elif task_wrenches is None:
        scenario_indices = sample_scenario_indices(
            scenario_key,
            table,
            split_id=split_id,
            count=num_perturbed,
            difficulty_bin=difficulty_bin,
            authority_regime=authority_regime,
            task_feasibility_regime=task_feasibility_regime,
            failure_type=failure_type,
            authority_label_any_mask=authority_label_any_mask,
        )
    else:
        scenario_indices = sample_task_conditioned_scenario_indices(
            scenario_key,
            table,
            split_id=split_id,
            task_wrenches=jnp.asarray(task_wrenches)[selected_envs],
            difficulty_bin=difficulty_bin,
            authority_regime=authority_regime,
            task_feasibility_regime=task_feasibility_regime,
            failure_type=failure_type,
            authority_label_any_mask=authority_label_any_mask,
        )
    if _STRICT_TIMING:
        jax.block_until_ready(scenario_indices)
    sample_duration = time.perf_counter() - timing_start
    timing_start = time.perf_counter()
    apply_detail = apply_failure_scenarios(
        env,
        key=apply_key,
        table=table,
        selected_envs=selected_envs,
        scenario_indices=scenario_indices,
        start_time=start_time,
    )
    if _STRICT_TIMING and Perturbation.thruster_mask is not None:
        jax.block_until_ready(Perturbation.thruster_mask)
    apply_duration = time.perf_counter() - timing_start
    timing_start = time.perf_counter()
    payload = scenario_selection_payload(table, scenario_indices)
    payload_duration = time.perf_counter() - timing_start
    payload["_timing/select"] = select_duration
    payload["_timing/sample"] = sample_duration
    payload["_timing/apply"] = apply_duration
    payload["_timing/payload"] = payload_duration
    payload["_timing/apply_disturbance"] = apply_detail["disturbance"]
    payload["_timing/apply_expand_scatter"] = apply_detail["expand_scatter"]
    payload["_timing/apply_state_build"] = apply_detail["state_build"]
    payload["_timing/apply_refresh"] = apply_detail["refresh"]
    if return_selection:
        return payload, selected_envs, scenario_indices
    return payload
