from dataclasses import dataclass
from enum import Enum
from typing import Optional, Tuple

import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np


@dataclass
class PerturbationState:
    rng: jnp.ndarray
    thruster_mask: jnp.ndarray
    failure_value: Optional[int] = None
    start_times: Optional[jnp.ndarray] = None
    max_thruster_force: Optional[jnp.ndarray] = None
    gp_x_samples: Optional[jnp.ndarray] = None
    gp_y_samples: Optional[jnp.ndarray] = None


def _perturbation_state_flatten(state: "PerturbationState"):
    children = (
        state.rng,
        state.thruster_mask,
        state.start_times,
        state.max_thruster_force,
        state.gp_x_samples,
        state.gp_y_samples,
    )
    return children, state.failure_value


def _perturbation_state_unflatten(aux_data, children):
    (
        rng,
        thruster_mask,
        start_times,
        max_thruster_force,
        gp_x_samples,
        gp_y_samples,
    ) = children
    return PerturbationState(
        rng=rng,
        thruster_mask=thruster_mask,
        failure_value=aux_data,
        start_times=start_times,
        max_thruster_force=max_thruster_force,
        gp_x_samples=gp_x_samples,
        gp_y_samples=gp_y_samples,
    )


jax.tree_util.register_pytree_node(
    PerturbationState,
    _perturbation_state_flatten,
    _perturbation_state_unflatten,
)


def _stable_cholesky(matrix: jnp.ndarray, base_jitter: float, max_attempts: int = 5):
    """Perform a Cholesky factorization, growing the jitter until the factor is finite."""
    dtype = matrix.dtype
    eye = jnp.eye(matrix.shape[0], dtype=dtype)
    jitter0 = jnp.asarray(base_jitter, dtype=dtype)

    def attempt(jitter):
        return jsp.linalg.cholesky(matrix + jitter * eye, lower=True)

    chol0 = attempt(jitter0)

    def cond_fn(state):
        step, jitter, chol = state
        bad = jnp.logical_not(jnp.isfinite(chol).all())
        return jnp.logical_and(bad, step < max_attempts)

    def body_fn(state):
        step, jitter, _ = state
        jitter = jitter * 10.0
        chol = attempt(jitter)
        return step + 1, jitter, chol

    init_state = (jnp.array(0, dtype=jnp.int32), jitter0, chol0)
    _, _, chol = jax.lax.while_loop(cond_fn, body_fn, init_state)
    return jnp.nan_to_num(chol)


def _derive_gp_resolution(num_thrusters: int, num_envs: int) -> Tuple[int, int]:
    """
    Mirror the classic perturbation setup: use more support points as the number of
    thrusters grows, and make the GP subset size depend on how many environments are
    being simulated in parallel.
    """
    num_points = max(32, min(256, 4 * max(1, num_thrusters)))
    # Scale subset size with the vectorized batch but keep it well-conditioned.
    subset_target = max(8, num_envs // 64 if num_envs > 0 else 8)
    subset_size = min(num_points, subset_target)
    if subset_size > num_points - 2:
        subset_size = max(2, num_points - 2)
    return num_points, subset_size


def perturbation_state_to_serializable(
    state: Optional["PerturbationState"],
) -> Optional[dict]:
    if state is None:
        return None

    to_np = lambda x: None if x is None else np.asarray(x)

    return {
        "rng": np.asarray(state.rng),
        "thruster_mask": np.asarray(state.thruster_mask),
        "failure_value": (
            None if state.failure_value is None else int(state.failure_value)
        ),
        "start_times": to_np(state.start_times),
        "max_thruster_force": to_np(state.max_thruster_force),
        "gp_x_samples": to_np(state.gp_x_samples),
        "gp_y_samples": to_np(state.gp_y_samples),
    }


def perturbation_state_from_serializable(
    payload: Optional[dict],
) -> Optional["PerturbationState"]:
    if payload is None:
        return None

    to_jnp = lambda x: None if x is None else jnp.asarray(x)

    return PerturbationState(
        rng=jnp.asarray(payload["rng"]),
        thruster_mask=jnp.asarray(payload["thruster_mask"]),
        failure_value=payload["failure_value"],
        start_times=to_jnp(payload["start_times"]),
        max_thruster_force=to_jnp(payload["max_thruster_force"]),
        gp_x_samples=to_jnp(payload["gp_x_samples"]),
        gp_y_samples=to_jnp(payload["gp_y_samples"]),
    )


def _broadcast_to_control_shape(arr: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Utility to broadcast stored perturbation arrays (which may be recorded per-env or per-thruster)
    to the full control array shape.
    """
    arr = jnp.asarray(arr)
    control = jnp.asarray(control)
    control_ndim = control.ndim
    if control_ndim == 0:
        return jnp.asarray(arr)
    if arr.shape == control.shape:
        return arr

    if arr.ndim == 0:
        arr = jnp.reshape(arr, (1,) * control_ndim)
    elif arr.ndim == 1:
        if arr.shape[0] == control.shape[-1]:
            arr = jnp.reshape(arr, (1, control.shape[-1]))
        elif arr.shape[0] == control.shape[0]:
            arr = jnp.reshape(arr, (control.shape[0], 1))
        else:
            arr = jnp.reshape(arr, (1,) * (control_ndim - 1) + arr.shape)
    elif arr.ndim == control_ndim - 1:
        if arr.shape[-1] == control.shape[-1]:
            arr = jnp.reshape(arr, (1,) + arr.shape)
        elif arr.shape[0] == control.shape[0]:
            arr = jnp.reshape(arr, arr.shape + (1,))
        else:
            arr = jnp.reshape(arr, (1,) * (control_ndim - arr.ndim) + arr.shape)
    elif arr.ndim > control_ndim:
        arr = jnp.reshape(arr, arr.shape[-control_ndim:])
    else:
        arr = jnp.reshape(arr, (1,) * (control_ndim - arr.ndim) + arr.shape)

    return jnp.broadcast_to(arr, control.shape)


def stuck_off_apply_from_state(
    state: Optional[PerturbationState],
    control: jnp.ndarray,
    timestamp: float,
) -> Tuple[jnp.ndarray, Optional[PerturbationState]]:
    if state is None:
        return control, state
    control = jnp.asarray(control)
    mask = jnp.asarray(state.thruster_mask)
    if mask.ndim == 1:
        if mask.shape[0] == control.shape[0]:
            mask = mask[:, None]
        elif mask.shape[0] == control.shape[-1]:
            mask = mask[None, :]
        else:
            mask = mask.reshape((1,) * (control.ndim - mask.ndim) + mask.shape)
    elif mask.ndim < control.ndim:
        mask = mask.reshape((1,) * (control.ndim - mask.ndim) + mask.shape)
    mask = jnp.broadcast_to(mask, control.shape)
    mask = mask == PerturbationStatus.STUCK_OFF.value

    if state.start_times is not None:
        start_times = _broadcast_to_control_shape(state.start_times, control)
    else:
        start_times = jnp.zeros(control.shape, dtype=control.dtype)

    timestamp_arr = _broadcast_to_control_shape(
        jnp.asarray(timestamp, dtype=start_times.dtype),
        control,
    )
    active = jnp.logical_and(mask, timestamp_arr >= start_times)
    control = jnp.where(active, 0.0, control)
    return control, state


def stuck_off_activate_state(
    state: Optional[PerturbationState],
    thruster_mask: jnp.ndarray,
    start_times: jnp.ndarray,
    rng: jnp.ndarray,
) -> PerturbationState:
    return PerturbationState(
        rng=rng,
        thruster_mask=thruster_mask,
        failure_value=PerturbationStatus.STUCK_OFF.value,
        start_times=start_times,
        max_thruster_force=state.max_thruster_force if state else None,
        gp_x_samples=state.gp_x_samples if state else None,
        gp_y_samples=state.gp_y_samples if state else None,
    )


def stuck_on_apply_from_state(
    state: Optional[PerturbationState],
    control: jnp.ndarray,
    timestamp: float,
) -> Tuple[jnp.ndarray, Optional[PerturbationState]]:
    if state is None:
        return control, state

    start_times = state.start_times
    max_force = state.max_thruster_force
    if start_times is None or max_force is None:
        return control, state

    control = jnp.asarray(control)
    mask = _broadcast_to_control_shape(state.thruster_mask, control)
    mask = mask == PerturbationStatus.STUCK_ON.value

    start_times = _broadcast_to_control_shape(start_times, control)

    timestamp_arr = _broadcast_to_control_shape(
        jnp.asarray(timestamp, dtype=start_times.dtype),
        control,
    )
    active = jnp.logical_and(mask, timestamp_arr >= start_times)
    broadcast_force = _broadcast_to_control_shape(max_force, control)
    control = jnp.where(active, broadcast_force, control)
    return control, state


def stuck_on_activate_state(
    state: Optional[PerturbationState],
    thruster_mask: jnp.ndarray,
    start_times: jnp.ndarray,
    max_thruster_force: jnp.ndarray,
    rng: jnp.ndarray,
) -> PerturbationState:
    return PerturbationState(
        rng=rng,
        thruster_mask=thruster_mask,
        failure_value=PerturbationStatus.STUCK_ON.value,
        start_times=start_times,
        max_thruster_force=max_thruster_force,
        gp_x_samples=state.gp_x_samples if state else None,
        gp_y_samples=state.gp_y_samples if state else None,
    )


def gp_apply_from_state(
    state: Optional[PerturbationState],
    control: jnp.ndarray,
    timestamp: float,
    failure_value: int,
) -> Tuple[jnp.ndarray, Optional[PerturbationState]]:
    if (
        state is None
        or state.start_times is None
        or state.gp_x_samples is None
        or state.gp_y_samples is None
    ):
        return control, state

    control = jnp.asarray(control)
    original_control = control
    gp_x_samples = state.gp_x_samples
    gp_y_samples = state.gp_y_samples

    mask = _broadcast_to_control_shape(state.thruster_mask, control)
    mask = mask == failure_value

    start_times = _broadcast_to_control_shape(state.start_times, control)

    timestamp_arr = _broadcast_to_control_shape(
        jnp.asarray(timestamp, dtype=start_times.dtype),
        control,
    )
    active = jnp.logical_and(mask, timestamp_arr >= start_times)

    gp_support = jnp.any(jnp.diff(gp_x_samples, axis=-1) != 0, axis=-1)
    gp_support = _broadcast_to_control_shape(gp_support, control)
    active = jnp.logical_and(active, gp_support)

    # Fast fixed-grid interpolation. GP registration/precomputation stores each
    # failed-thruster curve as a monotone lookup table, so runtime application
    # only needs two gathers and a linear blend instead of per-element jnp.interp.
    n_points = gp_y_samples.shape[-1]
    x_min = gp_x_samples[:, 0]
    x_max = gp_x_samples[:, -1]
    x_min_b = _broadcast_to_control_shape(x_min, control)
    x_max_b = _broadcast_to_control_shape(x_max, control)
    scale = (n_points - 1) / jnp.maximum(x_max_b - x_min_b, 1e-6)
    grid_pos = (control - x_min_b) * scale
    grid_pos = jnp.clip(grid_pos, 0.0, float(n_points - 1))
    idx0 = jnp.floor(grid_pos).astype(jnp.int32)
    idx1 = jnp.minimum(idx0 + 1, n_points - 1)
    frac = grid_pos - idx0.astype(grid_pos.dtype)

    if control.ndim == 1:
        thruster_idx = jnp.arange(control.shape[0], dtype=jnp.int32)
    else:
        thruster_idx = jnp.broadcast_to(
            jnp.arange(control.shape[-1], dtype=jnp.int32),
            control.shape,
        )
    y0 = gp_y_samples[thruster_idx, idx0]
    y1 = gp_y_samples[thruster_idx, idx1]
    interp_control = y0 + frac * (y1 - y0)
    interp_control = jnp.nan_to_num(
        interp_control,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    )
    control = jnp.where(active, interp_control, control)
    control = jnp.where(jnp.isnan(control), original_control, control)

    max_force = state.max_thruster_force
    if max_force is not None:
        # Temporary diagnostic clamp to ensure GP perturbations stay within the physical thrust envelope.
        broadcast_force = _broadcast_to_control_shape(max_force, control)
        control = jnp.clip(control, -broadcast_force, broadcast_force)

    return control, state


def gp_register_state(
    state: Optional[PerturbationState],
    thruster_mask: jnp.ndarray,
    start_times: jnp.ndarray,
    rng: jnp.ndarray,
    failure_value: int,
    thruster_index: int,
    x_samples: jnp.ndarray,
    y_samples: jnp.ndarray,
) -> PerturbationState:
    num_thrusters = thruster_mask.shape[1]
    num_points = x_samples.shape[0]
    sanitize = lambda arr: jnp.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    x_samples = sanitize(x_samples)
    y_samples = sanitize(y_samples)

    if state is not None and state.gp_x_samples is not None:
        gp_x = state.gp_x_samples
        gp_y = state.gp_y_samples
    else:
        gp_x = jnp.zeros((num_thrusters, num_points), dtype=x_samples.dtype)
        gp_y = jnp.zeros_like(gp_x)

    gp_x = gp_x.at[thruster_index].set(x_samples)
    gp_y = gp_y.at[thruster_index].set(y_samples)
    gp_x = sanitize(gp_x)
    gp_y = sanitize(gp_y)

    return PerturbationState(
        rng=rng,
        thruster_mask=thruster_mask,
        failure_value=failure_value,
        start_times=start_times,
        max_thruster_force=state.max_thruster_force if state else None,
        gp_x_samples=gp_x,
        gp_y_samples=gp_y,
    )


class PerturbationStatus(Enum):
    """
    Document type of active perturbation.
    0 := Thruster fully operational
    1 := Thruster is stuck off
    2 := Thruster is stuck on
    3 := Thruster valve is faulty
    4 := Thruster output is saturated
    5 := Thruster output is unstable
    """

    OPERATIONAL = 0
    STUCK_OFF = 1
    STUCK_ON = 2
    FAULTY_VALVE = 3
    SATURATED_THRUST = 4
    THRUST_INSTABILITY = 5
