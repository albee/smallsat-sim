from dataclasses import dataclass

import jax
import jax.numpy as jnp
import pytest

from smallsat_sim.envs import vec_env
from smallsat_sim.envs.perturbations_rl import (
    PerturbationState,
    PerturbationStatus,
    gp_apply_from_state,
    stuck_off_apply_from_state,
    stuck_on_apply_from_state,
)


@dataclass
class DummyBatch:
    qfrc_applied: jnp.ndarray
    time: jnp.ndarray


def _build_state(
    *,
    failure_value: int,
    thruster_mask_value: int,
    num_envs: int,
    num_thrusters: int,
) -> PerturbationState:
    thruster_mask = jnp.full(
        (num_envs, num_thrusters), thruster_mask_value, dtype=jnp.int32
    )
    start_times = jnp.zeros((num_envs, num_thrusters), dtype=jnp.float32)
    gp_x_samples = jnp.tile(
        jnp.array([0.0, 0.5, 1.0], dtype=jnp.float32), (num_thrusters, 1)
    )
    gp_y_samples = jnp.tile(
        jnp.array([0.0, 0.2, 0.8], dtype=jnp.float32), (num_thrusters, 1)
    )
    max_thruster_force = jnp.full(
        (num_envs, num_thrusters), 1.0, dtype=jnp.float32
    )
    return PerturbationState(
        rng=jax.random.PRNGKey(0),
        thruster_mask=thruster_mask,
        failure_value=failure_value,
        start_times=start_times,
        max_thruster_force=max_thruster_force,
        gp_x_samples=gp_x_samples,
        gp_y_samples=gp_y_samples,
    )


@pytest.mark.parametrize(
    "failure_value,expected_fn",
    [
        (PerturbationStatus.STUCK_OFF.value, stuck_off_apply_from_state),
        (PerturbationStatus.STUCK_ON.value, stuck_on_apply_from_state),
        (PerturbationStatus.FAULTY_VALVE.value, gp_apply_from_state),
        (PerturbationStatus.SATURATED_THRUST.value, gp_apply_from_state),
        (PerturbationStatus.THRUST_INSTABILITY.value, gp_apply_from_state),
    ],
)
def test_prepare_step_functional_routes_failure_modes(
    failure_value: int, expected_fn
) -> None:
    num_envs = 1
    num_thrusters = 2
    base_ctrl = jnp.array([[0.1, 0.9]], dtype=jnp.float32)
    dummy_batch = DummyBatch(
        qfrc_applied=jnp.zeros_like(base_ctrl),
        time=jnp.array([1.0], dtype=jnp.float32),
    )
    pert_state = _build_state(
        failure_value=failure_value,
        thruster_mask_value=failure_value,
        num_envs=num_envs,
        num_thrusters=num_thrusters,
    )
    env_state = vec_env.VecEnvState(
        rng=jax.random.PRNGKey(1),
        mjx_batch=dummy_batch,
        terminal_hold_counts=jnp.zeros((num_envs,), dtype=jnp.int32),
        disturbance_states=(),
        perturbation_states=(pert_state,),
    )

    _, applied_ctrl, _ = vec_env.prepare_step_functional(env_state, base_ctrl)

    if expected_fn is gp_apply_from_state:
        expected_ctrl, _ = expected_fn(
            pert_state, base_ctrl, dummy_batch.time, failure_value
        )
    else:
        expected_ctrl, _ = expected_fn(pert_state, base_ctrl, dummy_batch.time)
    assert jnp.allclose(applied_ctrl, expected_ctrl)


def test_prepare_step_functional_fallback_applies_masked_stuck_off() -> None:
    num_envs = 1
    num_thrusters = 2
    base_ctrl = jnp.array([[0.3, 0.7]], dtype=jnp.float32)
    dummy_batch = DummyBatch(
        qfrc_applied=jnp.zeros_like(base_ctrl),
        time=jnp.array([1.0], dtype=jnp.float32),
    )
    pert_state = _build_state(
        failure_value=999,
        thruster_mask_value=PerturbationStatus.STUCK_OFF.value,
        num_envs=num_envs,
        num_thrusters=num_thrusters,
    )
    env_state = vec_env.VecEnvState(
        rng=jax.random.PRNGKey(2),
        mjx_batch=dummy_batch,
        terminal_hold_counts=jnp.zeros((num_envs,), dtype=jnp.int32),
        disturbance_states=(),
        perturbation_states=(pert_state,),
    )

    _, applied_ctrl, _ = vec_env.prepare_step_functional(env_state, base_ctrl)

    expected_ctrl, _ = stuck_off_apply_from_state(
        pert_state, base_ctrl, dummy_batch.time
    )
    assert jnp.allclose(applied_ctrl, expected_ctrl)
