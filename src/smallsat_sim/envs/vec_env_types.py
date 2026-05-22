from dataclasses import dataclass, replace
from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from mujoco import mjx

from smallsat_sim.envs.disturbances import (
    DisturbanceState,
    disturbance_state_from_serializable,
    disturbance_state_to_serializable,
)
from smallsat_sim.envs.perturbations_rl import (
    PerturbationState,
    perturbation_state_from_serializable,
    perturbation_state_to_serializable,
)


@dataclass
class VecEnvState:
    """
    Minimal container for the mutable pieces of the vectorised environment.

    Storing the RNG key alongside the MJX batch makes it easier to convert the
    environment into a pure functional representation in later steps.
    """

    rng: jnp.ndarray
    mjx_batch: mjx.Data
    terminal_hold_counts: jnp.ndarray
    disturbance_states: Tuple[Optional[DisturbanceState], ...] = ()
    perturbation_states: Tuple[Optional[PerturbationState], ...] = ()

    def replace(self, **updates) -> "VecEnvState":
        return replace(self, **updates)


@dataclass
class FreeFlyerVecEnvState:
    """
    Compact free-flyer rollout state for fast policy-training scans.

    This intentionally carries only the dynamic state needed by the RL reward,
    terminal logic, perturbations, and residual update. MJX remains available for
    reset/evaluation/deployment; this state is only an optional rollout backend.
    """

    rng: jnp.ndarray
    qpos: jnp.ndarray
    vel_body: jnp.ndarray
    omega: jnp.ndarray
    time: jnp.ndarray
    ctrl: jnp.ndarray
    actuator_force: jnp.ndarray
    terminal_hold_counts: jnp.ndarray
    disturbance_states: Tuple[Optional[DisturbanceState], ...] = ()
    perturbation_states: Tuple[Optional[PerturbationState], ...] = ()

    def replace(self, **updates) -> "FreeFlyerVecEnvState":
        return replace(self, **updates)


@dataclass(eq=False)
class VecEnvStepConfig:
    """
    Static configuration required to evolve the vectorised environment in a pure way.

    Invariants:
    - All arrays are broadcast across `num_envs` environments.
    - `control_decimation` mirrors the imperative `VecEnv.step` loop, so the helper
      behaviour matches the legacy path.
    - `base_disturbance_states` / `base_perturbation_states` capture mutable effect
      snapshots that must be refreshed whenever the environment is reset inside a scan.
    """

    mjx_model: mjx.Model
    mjx_data_template: mjx.Data
    mjx_batch_template: mjx.Data
    init_qpos: jnp.ndarray
    init_qvel: jnp.ndarray
    num_envs: int
    max_start_offset: float
    control_decimation: int
    sigma_pos: float
    sigma_vel: float
    sigma_att: float
    sigma_angvel: float
    w_pos: float
    w_vel: float
    w_att: float
    w_angvel: float
    lam_fuel: float
    lam_speed_terminal: float
    lam_ang_speed_terminal: float
    lam_fuel_terminal: float
    terminal_bonus: float
    terminal_radius: float
    terminal_max_speed: float
    terminal_max_att_error: float
    terminal_max_ang_speed: float
    lam_wrench_residual: float
    wrench_residual_tolerance: float
    wrench_residual_clip: float
    max_episode_len: int
    terminal_hold_steps: int
    enable_failure_termination: bool
    failure_max_position_error: float
    failure_max_speed: float
    failure_max_att_error: float
    failure_max_ang_speed: float
    res_dim: int
    use_adaptive_approach: bool
    collect_reward_components: bool
    thruster_mixer_T: jnp.ndarray
    base_disturbance_states: Tuple[Optional[DisturbanceState], ...] = ()
    base_perturbation_states: Tuple[Optional[PerturbationState], ...] = ()
    mass: float = 1.0
    inertia_diag: Optional[jnp.ndarray] = None
    model_dt: float = 0.01
    effects_enabled: bool = True


@dataclass
class VecEnvStepOutput:
    """
    Container for the results of a pure VecEnv step. Keeping this as a PyTree allows
    the rollout to be scanned/`jax.jit`ed later on.
    """

    prev_states: jnp.ndarray
    next_states: jnp.ndarray
    rewards: jnp.ndarray
    terminals: jnp.ndarray
    commanded_ctrl: jnp.ndarray
    applied_ctrl: jnp.ndarray
    actual_wrench: jnp.ndarray
    desired_wrench: jnp.ndarray
    prev_obs: jnp.ndarray
    next_obs: jnp.ndarray
    success_terminals: jnp.ndarray
    failure_terminals: jnp.ndarray
    reward_components: Dict[str, jnp.ndarray]


@dataclass
class VecEnvTrainingStepOutput:
    """
    Lean transition record for PPO policy training.

    It intentionally omits absolute observations and reward-component dictionaries
    to reduce scan output size and device memory traffic in the hot training loop.
    """

    prev_states: jnp.ndarray
    next_position_error: jnp.ndarray
    rewards: jnp.ndarray
    terminals: jnp.ndarray
    applied_ctrl: jnp.ndarray
    actual_wrench: jnp.ndarray
    success_terminals: jnp.ndarray
    failure_terminals: jnp.ndarray


def _vecenv_step_output_flatten(output: VecEnvStepOutput):
    children = (
        output.prev_states,
        output.next_states,
        output.rewards,
        output.terminals,
        output.commanded_ctrl,
        output.applied_ctrl,
        output.actual_wrench,
        output.desired_wrench,
        output.prev_obs,
        output.next_obs,
        output.success_terminals,
        output.failure_terminals,
        output.reward_components,
    )
    return children, None


def _vecenv_step_output_unflatten(aux_data, children):
    (
        prev_states,
        next_states,
        rewards,
        terminals,
        commanded_ctrl,
        applied_ctrl,
        actual_wrench,
        desired_wrench,
        prev_obs,
        next_obs,
        success_terminals,
        failure_terminals,
        reward_components,
    ) = children
    return VecEnvStepOutput(
        prev_states=prev_states,
        next_states=next_states,
        rewards=rewards,
        terminals=terminals,
        commanded_ctrl=commanded_ctrl,
        applied_ctrl=applied_ctrl,
        actual_wrench=actual_wrench,
        desired_wrench=desired_wrench,
        prev_obs=prev_obs,
        next_obs=next_obs,
        success_terminals=success_terminals,
        failure_terminals=failure_terminals,
        reward_components=reward_components,
    )


jax.tree_util.register_pytree_node(
    VecEnvStepOutput,
    _vecenv_step_output_flatten,
    _vecenv_step_output_unflatten,
)


def _vecenv_training_step_output_flatten(output: VecEnvTrainingStepOutput):
    children = (
        output.prev_states,
        output.next_position_error,
        output.rewards,
        output.terminals,
        output.applied_ctrl,
        output.actual_wrench,
        output.success_terminals,
        output.failure_terminals,
    )
    return children, None


def _vecenv_training_step_output_unflatten(aux_data, children):
    (
        prev_states,
        next_position_error,
        rewards,
        terminals,
        applied_ctrl,
        actual_wrench,
        success_terminals,
        failure_terminals,
    ) = children
    return VecEnvTrainingStepOutput(
        prev_states=prev_states,
        next_position_error=next_position_error,
        rewards=rewards,
        terminals=terminals,
        applied_ctrl=applied_ctrl,
        actual_wrench=actual_wrench,
        success_terminals=success_terminals,
        failure_terminals=failure_terminals,
    )


jax.tree_util.register_pytree_node(
    VecEnvTrainingStepOutput,
    _vecenv_training_step_output_flatten,
    _vecenv_training_step_output_unflatten,
)


def _vecenv_state_flatten(state: "VecEnvState"):
    children = (
        state.rng,
        state.mjx_batch,
        state.terminal_hold_counts,
        state.disturbance_states,
        state.perturbation_states,
    )
    return children, None


def _vecenv_state_unflatten(aux_data, children):
    rng, mjx_batch, terminal_hold_counts, disturbance_states, perturbation_states = children
    return VecEnvState(
        rng=rng,
        mjx_batch=mjx_batch,
        terminal_hold_counts=terminal_hold_counts,
        disturbance_states=tuple(disturbance_states),
        perturbation_states=tuple(perturbation_states),
    )


jax.tree_util.register_pytree_node(
    VecEnvState,
    _vecenv_state_flatten,
    _vecenv_state_unflatten,
)


def _freeflyer_state_flatten(state: "FreeFlyerVecEnvState"):
    children = (
        state.rng,
        state.qpos,
        state.vel_body,
        state.omega,
        state.time,
        state.ctrl,
        state.actuator_force,
        state.terminal_hold_counts,
        state.disturbance_states,
        state.perturbation_states,
    )
    return children, None


def _freeflyer_state_unflatten(aux_data, children):
    (
        rng,
        qpos,
        vel_body,
        omega,
        time_arr,
        ctrl,
        actuator_force,
        terminal_hold_counts,
        disturbance_states,
        perturbation_states,
    ) = children
    return FreeFlyerVecEnvState(
        rng=rng,
        qpos=qpos,
        vel_body=vel_body,
        omega=omega,
        time=time_arr,
        ctrl=ctrl,
        actuator_force=actuator_force,
        terminal_hold_counts=terminal_hold_counts,
        disturbance_states=tuple(disturbance_states),
        perturbation_states=tuple(perturbation_states),
    )


jax.tree_util.register_pytree_node(
    FreeFlyerVecEnvState,
    _freeflyer_state_flatten,
    _freeflyer_state_unflatten,
)


def vecenv_state_to_serializable(state: "VecEnvState") -> dict:
    """
    Convert a VecEnvState into host-serializable numpy-backed payload.
    """

    def to_numpy(value):
        if value is None:
            return None
        if isinstance(value, jnp.ndarray):
            return np.asarray(value)
        if isinstance(value, np.ndarray):
            return value
        return value

    mjx_fields = getattr(state.mjx_batch, "__dataclass_fields__", {})
    mjx_payload = {}
    for name in mjx_fields:
        mjx_payload[name] = to_numpy(getattr(state.mjx_batch, name))

    return {
        "rng": np.asarray(state.rng),
        "mjx": mjx_payload,
        "terminal_hold_counts": np.asarray(state.terminal_hold_counts),
        "disturbance_states": [
            disturbance_state_to_serializable(s) for s in state.disturbance_states
        ],
        "perturbation_states": [
            perturbation_state_to_serializable(s) for s in state.perturbation_states
        ],
    }


def vecenv_state_from_serializable(
    payload: dict,
    *,
    mjx_batch_template: mjx.Data,
) -> "VecEnvState":
    """
    Reconstruct a VecEnvState from the serialized payload using the provided mjx
    template (typically the environment's freshly reset batch).
    """

    def to_jnp(value):
        if value is None:
            return None
        if isinstance(value, np.ndarray):
            return jnp.asarray(value)
        if isinstance(value, jnp.ndarray):
            return value
        return value

    mjx_updates = {name: to_jnp(value) for name, value in payload["mjx"].items()}
    mjx_batch = mjx_batch_template.replace(**mjx_updates)

    disturbance_states = tuple(
        disturbance_state_from_serializable(s) if s is not None else None
        for s in payload["disturbance_states"]
    )
    perturbation_states = tuple(
        perturbation_state_from_serializable(s) if s is not None else None
        for s in payload["perturbation_states"]
    )

    return VecEnvState(
        rng=jnp.asarray(payload["rng"]),
        mjx_batch=mjx_batch,
        terminal_hold_counts=jnp.asarray(
            payload.get(
                "terminal_hold_counts",
                np.zeros((mjx_batch.qpos.shape[0],), dtype=np.int32),
            )
        ),
        disturbance_states=disturbance_states,
        perturbation_states=perturbation_states,
    )
