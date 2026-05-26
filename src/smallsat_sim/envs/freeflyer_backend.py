from typing import Optional, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from smallsat_sim.envs.disturbances import (
    DisturbanceState,
    constant_force_apply_from_state,
)
from smallsat_sim.envs.perturbations_rl import (
    PerturbationState,
    PerturbationStatus,
    gp_apply_from_state,
    stuck_off_apply_from_state,
    stuck_on_apply_from_state,
)
from smallsat_sim.envs.vec_env_common import (
    _compute_failures,
    _compute_penalties,
    _compute_terminals,
    _quat_log_error,
    _quat_t_batch,
    _quat_to_rot_batch,
    _reward_components,
    _sample_random_quat,
)
from smallsat_sim.envs.vec_env_types import (
    FreeFlyerVecEnvState,
    VecEnvState,
    VecEnvStepConfig,
    VecEnvStepOutput,
    VecEnvTrainingStepOutput,
)


def _compute_freeflyer_state_features(
    state: FreeFlyerVecEnvState,
    next_waypoint: jnp.ndarray,
) -> jnp.ndarray:
    num_envs = state.qpos.shape[0]
    reference = jnp.asarray(next_waypoint)
    if reference.ndim == 1:
        reference = jnp.broadcast_to(reference, (num_envs, reference.shape[0]))

    delta_pos = state.qpos[:, :3] - reference[:, :3]
    attitude_error = jax.vmap(_quat_log_error)(state.qpos[:, 3:7], reference[:, 3:7])
    return jnp.concatenate(
        (delta_pos, attitude_error, state.vel_body, state.omega),
        axis=1,
    )


def _compute_freeflyer_observations(state: FreeFlyerVecEnvState) -> jnp.ndarray:
    return jnp.concatenate((state.qpos, state.vel_body, state.omega), axis=1)


def _freeflyer_from_mjx_state(state: VecEnvState) -> FreeFlyerVecEnvState:
    rot_world_to_body = jnp.swapaxes(state.mjx_batch.xmat[:, 1, :, :], 1, 2)
    vel_body = jnp.einsum("bij,bj->bi", rot_world_to_body, state.mjx_batch.qvel[:, :3])
    return FreeFlyerVecEnvState(
        rng=state.rng,
        qpos=state.mjx_batch.qpos,
        vel_body=vel_body,
        omega=state.mjx_batch.qvel[:, 3:6],
        time=state.mjx_batch.time,
        ctrl=state.mjx_batch.ctrl,
        actuator_force=state.mjx_batch.actuator_force,
        terminal_hold_counts=state.terminal_hold_counts,
        disturbance_states=state.disturbance_states,
        perturbation_states=state.perturbation_states,
    )


def freeflyer_to_mjx_state(
    state: FreeFlyerVecEnvState,
    template_state: VecEnvState,
    config: VecEnvStepConfig,
) -> VecEnvState:
    rot_body_to_world = _quat_to_rot_batch(state.qpos[:, 3:7])
    vel_world = jnp.einsum("bij,bj->bi", rot_body_to_world, state.vel_body)
    qvel = jnp.concatenate((vel_world, state.omega), axis=1)
    mjx_batch = template_state.mjx_batch.replace(
        qpos=state.qpos,
        qvel=qvel,
        time=state.time,
        ctrl=state.ctrl,
        actuator_force=state.actuator_force,
    )
    mjx_batch = jax.vmap(mjx.forward, in_axes=(None, 0))(config.mjx_model, mjx_batch)
    mjx_batch = mjx_batch.replace(
        time=state.time,
        ctrl=state.ctrl,
        actuator_force=state.actuator_force,
    )
    return VecEnvState(
        rng=state.rng,
        mjx_batch=mjx_batch,
        terminal_hold_counts=state.terminal_hold_counts,
        disturbance_states=state.disturbance_states,
        perturbation_states=state.perturbation_states,
    )


def freeflyer_reset(
    rng_key: jnp.ndarray,
    *,
    config: VecEnvStepConfig,
) -> FreeFlyerVecEnvState:
    """
    Compact reset matching vecenv_reset's position/attitude randomization.

    This avoids mjx.forward during free-flyer policy-training rollouts. It is
    valid because the compact backend only needs qpos, body velocity, angular
    velocity, controls, time, and terminal counters.
    """
    rng_key, split_key = jax.random.split(rng_key)
    env_keys = jax.random.split(split_key, config.num_envs)
    base_qpos = config.init_qpos[0]
    base_qvel = config.init_qvel[0]

    def randomize_state(env_key):
        pos_key, quat_key, linvel_key, angvel_key = jax.random.split(env_key, 4)
        u = jax.random.uniform(pos_key, (2,))
        r = config.max_start_offset * jnp.sqrt(u[0])
        theta = 2.0 * jnp.pi * u[1]
        random_pos = jnp.array(
            [
                base_qpos[0] + r * jnp.cos(theta),
                base_qpos[1] + r * jnp.sin(theta),
                base_qpos[2],
            ],
            dtype=base_qpos.dtype,
        )
        random_quat = _sample_random_quat(quat_key).astype(base_qpos.dtype)
        linvel = jax.random.uniform(
            linvel_key,
            (3,),
            minval=-float(config.max_start_linear_velocity),
            maxval=float(config.max_start_linear_velocity),
        ).astype(base_qvel.dtype)
        angvel = jax.random.uniform(
            angvel_key,
            (3,),
            minval=-float(config.max_start_angular_velocity),
            maxval=float(config.max_start_angular_velocity),
        ).astype(base_qvel.dtype)
        qvel = base_qvel
        if base_qvel.shape[0] >= 3:
            qvel = qvel.at[:3].set(linvel)
        if base_qvel.shape[0] >= 6:
            qvel = qvel.at[3:6].set(angvel)
        return jnp.concatenate([random_pos, random_quat]), qvel

    qpos, qvel = jax.vmap(randomize_state)(env_keys)
    return FreeFlyerVecEnvState(
        rng=rng_key,
        qpos=qpos,
        vel_body=qvel[:, :3],
        omega=qvel[:, 3:6],
        time=jnp.zeros((config.num_envs,), dtype=qpos.dtype),
        ctrl=jnp.zeros(
            (config.num_envs, config.thruster_mixer_T.shape[0]),
            dtype=qpos.dtype,
        ),
        actuator_force=jnp.zeros(
            (config.num_envs, config.thruster_mixer_T.shape[0]),
            dtype=qpos.dtype,
        ),
        terminal_hold_counts=jnp.zeros((config.num_envs,), dtype=jnp.int32),
        disturbance_states=config.base_disturbance_states,
        perturbation_states=config.base_perturbation_states,
    )


def freeflyer_reset_masked(
    state: FreeFlyerVecEnvState,
    config: VecEnvStepConfig,
    reset_mask: jnp.ndarray,
) -> FreeFlyerVecEnvState:
    reset_state = freeflyer_reset(state.rng, config=config)

    def merge(reset_leaf, current_leaf):
        if not hasattr(reset_leaf, "shape") or len(reset_leaf.shape) == 0:
            return current_leaf
        if reset_leaf.shape[0] != config.num_envs:
            return current_leaf
        mask_shape = (config.num_envs,) + (1,) * (len(reset_leaf.shape) - 1)
        return jnp.where(reset_mask.reshape(mask_shape), reset_leaf, current_leaf)

    return FreeFlyerVecEnvState(
        rng=reset_state.rng,
        qpos=merge(reset_state.qpos, state.qpos),
        vel_body=merge(reset_state.vel_body, state.vel_body),
        omega=merge(reset_state.omega, state.omega),
        time=merge(reset_state.time, state.time),
        ctrl=merge(reset_state.ctrl, state.ctrl),
        actuator_force=merge(reset_state.actuator_force, state.actuator_force),
        terminal_hold_counts=jnp.where(
            reset_mask,
            jnp.zeros_like(state.terminal_hold_counts),
            state.terminal_hold_counts,
        ),
        disturbance_states=state.disturbance_states,
        perturbation_states=state.perturbation_states,
    )

def _prepare_freeflyer_step(
    state: FreeFlyerVecEnvState,
    base_ctrl: jnp.ndarray,
) -> Tuple[FreeFlyerVecEnvState, jnp.ndarray, jnp.ndarray]:
    ctrl = base_ctrl
    qfrc_applied = jnp.zeros((state.qpos.shape[0], 6), dtype=base_ctrl.dtype)
    updated_disturbances: list[Optional[DisturbanceState]] = []
    for dist_state in state.disturbance_states:
        if dist_state is None:
            updated_disturbances.append(None)
            continue
        const_force = dist_state.params.get("const_force") if dist_state.params else None
        if const_force is not None:
            force_contrib, new_state = constant_force_apply_from_state(
                dist_state,
                state.time,
                const_force,
            )
            qfrc_applied = qfrc_applied + force_contrib[:, :6]
            updated_disturbances.append(new_state)
        else:
            updated_disturbances.append(dist_state)

    updated_perturbations: list[Optional[PerturbationState]] = []
    for pert_state in state.perturbation_states:
        if pert_state is None:
            updated_perturbations.append(None)
            continue

        new_state = pert_state
        failure_value = getattr(new_state, "failure_value", None)
        if failure_value is None:
            failure_value_arr = jnp.asarray(-1, dtype=jnp.int32)
        else:
            failure_value_arr = jnp.asarray(failure_value)

        is_stuck_off = failure_value_arr == PerturbationStatus.STUCK_OFF.value
        is_stuck_on = failure_value_arr == PerturbationStatus.STUCK_ON.value
        is_gp = jnp.logical_or(
            failure_value_arr == PerturbationStatus.FAULTY_VALVE.value,
            jnp.logical_or(
                failure_value_arr == PerturbationStatus.SATURATED_THRUST.value,
                failure_value_arr == PerturbationStatus.THRUST_INSTABILITY.value,
            ),
        )
        mode = jnp.where(
            is_stuck_off,
            jnp.array(0, dtype=jnp.int32),
            jnp.where(
                is_stuck_on,
                jnp.array(1, dtype=jnp.int32),
                jnp.where(is_gp, jnp.array(2, dtype=jnp.int32), jnp.array(3, dtype=jnp.int32)),
            ),
        )

        def _branch_stuck_off(operand):
            state_in, ctrl_in, time_in, _ = operand
            ctrl_out, state_out = stuck_off_apply_from_state(state_in, ctrl_in, time_in)
            return state_out, ctrl_out

        def _branch_stuck_on(operand):
            state_in, ctrl_in, time_in, _ = operand
            ctrl_out, state_out = stuck_on_apply_from_state(state_in, ctrl_in, time_in)
            return state_out, ctrl_out

        def _branch_gp(operand):
            state_in, ctrl_in, time_in, failure_in = operand
            ctrl_out, state_out = gp_apply_from_state(
                state_in,
                ctrl_in,
                time_in,
                failure_in,
            )
            return state_out, ctrl_out

        def _branch_fallback(operand):
            state_in, ctrl_in, time_in, _ = operand
            ctrl_mid, state_mid = stuck_off_apply_from_state(state_in, ctrl_in, time_in)
            ctrl_out, state_out = stuck_on_apply_from_state(state_mid, ctrl_mid, time_in)
            return state_out, ctrl_out

        new_state, ctrl = jax.lax.switch(
            mode,
            (_branch_stuck_off, _branch_stuck_on, _branch_gp, _branch_fallback),
            (new_state, ctrl, state.time, failure_value_arr),
        )
        updated_perturbations.append(new_state)

    return (
        state.replace(
            disturbance_states=tuple(updated_disturbances),
            perturbation_states=tuple(updated_perturbations),
        ),
        ctrl,
        qfrc_applied,
    )


def _freeflyer_step_dynamics(
    state: FreeFlyerVecEnvState,
    ctrl: jnp.ndarray,
    qfrc_applied: jnp.ndarray,
    config: VecEnvStepConfig,
) -> FreeFlyerVecEnvState:
    wrench = ctrl @ config.thruster_mixer_T
    rot_body_to_world = _quat_to_rot_batch(state.qpos[:, 3:7])
    rot_world_to_body = jnp.swapaxes(rot_body_to_world, 1, 2)
    force_body = wrench[:, :3] + jnp.einsum(
        "bij,bj->bi", rot_world_to_body, qfrc_applied[:, :3]
    )
    torque_body = wrench[:, 3:6] + jnp.einsum(
        "bij,bj->bi", rot_world_to_body, qfrc_applied[:, 3:6]
    )
    quat_t = _quat_t_batch(state.qpos[:, 3:7])
    dt = jnp.asarray(config.model_dt, dtype=state.qpos.dtype)
    mass = jnp.asarray(config.mass, dtype=state.qpos.dtype)
    inertia = jnp.asarray(config.inertia_diag, dtype=state.qpos.dtype)

    pos_dot = jnp.einsum("bij,bj->bi", rot_body_to_world, state.vel_body)
    quat_dot = jnp.einsum("bij,bj->bi", quat_t, state.omega)
    vel_dot = force_body / mass - jnp.cross(state.omega, state.vel_body)
    inertia_omega = state.omega * inertia
    omega_dot = (torque_body - jnp.cross(state.omega, inertia_omega)) / inertia

    qpos_next = jnp.concatenate(
        (
            state.qpos[:, :3] + dt * pos_dot,
            state.qpos[:, 3:7] + dt * quat_dot,
        ),
        axis=1,
    )
    quat_next = qpos_next[:, 3:7]
    quat_next = quat_next / (jnp.linalg.norm(quat_next, axis=1, keepdims=True) + 1e-9)
    qpos_next = qpos_next.at[:, 3:7].set(quat_next)
    return state.replace(
        qpos=qpos_next,
        vel_body=state.vel_body + dt * vel_dot,
        omega=state.omega + dt * omega_dot,
        time=state.time + dt,
        ctrl=ctrl,
        actuator_force=ctrl,
    )


def vecenv_step_training_freeflyer(
    state: FreeFlyerVecEnvState,
    commanded_ctrl: jnp.ndarray,
    next_waypoint: jnp.ndarray,
    config: VecEnvStepConfig,
    prev_residuals: Optional[jnp.ndarray] = None,
    prev_states: Optional[jnp.ndarray] = None,
    return_next_states: bool = False,
) -> Tuple[FreeFlyerVecEnvState, VecEnvTrainingStepOutput]:
    commanded_ctrl = jnp.asarray(commanded_ctrl)

    if prev_states is None:
        prev_states = _compute_freeflyer_state_features(state, next_waypoint)
    if config.effects_enabled:
        prepared_state, applied_ctrl, qfrc_applied = _prepare_freeflyer_step(
            state, commanded_ctrl
        )
    else:
        prepared_state = state
        applied_ctrl = commanded_ctrl
        qfrc_applied = jnp.zeros((state.qpos.shape[0], 6), dtype=commanded_ctrl.dtype)

    next_state = jax.lax.fori_loop(
        0,
        config.control_decimation,
        lambda _i, sub_state: _freeflyer_step_dynamics(
            sub_state,
            applied_ctrl,
            qfrc_applied,
            config,
        ),
        prepared_state,
    )
    next_states = _compute_freeflyer_state_features(next_state, next_waypoint)

    pos_curr, vel_curr, att_curr, ang_curr = _reward_components(prev_states, config)
    pos_next, vel_next, att_next, ang_next = _reward_components(next_states, config)
    rewards = (pos_next + vel_next + att_next + ang_next) - (
        pos_curr + vel_curr + att_curr + ang_curr
    )

    success_terminals, next_terminal_hold_counts = _compute_terminals(
        next_states,
        prepared_state.terminal_hold_counts,
        config,
    )
    failure_terminals = _compute_failures(next_states, config)
    terminals = jnp.logical_or(success_terminals, failure_terminals)

    if config.use_adaptive_approach and config.res_dim > 0:
        if prev_residuals is None or prev_residuals.shape[-1] == 0:
            prev_residuals = (
                jnp.atleast_2d(state.actuator_force) @ config.thruster_mixer_T
                - jnp.atleast_2d(state.ctrl) @ config.thruster_mixer_T
            )
        else:
            prev_residuals = jnp.asarray(prev_residuals, dtype=prev_states.dtype)
    else:
        prev_residuals = None

    penalties, penalty_components = _compute_penalties(
        prev_states,
        next_states,
        success_terminals,
        commanded_ctrl,
        prev_residuals,
        config,
    )
    terminal_bonus = jnp.asarray(config.terminal_bonus, dtype=rewards.dtype)
    rewards = rewards - penalties + terminal_bonus * success_terminals.astype(rewards.dtype)

    if config.use_adaptive_approach and config.res_dim > 0:
        actual_wrench = jnp.atleast_2d(applied_ctrl) @ config.thruster_mixer_T
    else:
        wrench_shape = (commanded_ctrl.shape[0], config.thruster_mixer_T.shape[1])
        actual_wrench = jnp.zeros(wrench_shape, dtype=commanded_ctrl.dtype)

    if config.collect_reward_components:
        shaping_deltas = jnp.stack(
            [
                pos_next - pos_curr,
                vel_next - vel_curr,
                att_next - att_curr,
                ang_next - ang_curr,
            ],
            axis=1,
        )
        reward_components = {
            "shaping_pos": shaping_deltas[:, 0],
            "shaping_vel": shaping_deltas[:, 1],
            "shaping_att": shaping_deltas[:, 2],
            "shaping_angvel": shaping_deltas[:, 3],
            "shaping_total": shaping_deltas.sum(axis=1),
            **penalty_components,
            "bonus_terminal": terminal_bonus
            * success_terminals.astype(rewards.dtype),
            "terminated_success": success_terminals.astype(rewards.dtype),
            "terminated_failure": failure_terminals.astype(rewards.dtype),
            "reward_total": rewards,
        }
    else:
        reward_components = {}

    step_output = VecEnvTrainingStepOutput(
        prev_states=prev_states,
        next_position_error=next_states[:, :3],
        next_attitude_error=jnp.linalg.norm(next_states[:, 3:6], axis=1),
        next_speed=jnp.linalg.norm(next_states[:, 6:9], axis=1),
        next_angular_speed=jnp.linalg.norm(next_states[:, 9:12], axis=1),
        rewards=rewards,
        terminals=terminals,
        applied_ctrl=applied_ctrl,
        actual_wrench=actual_wrench,
        success_terminals=success_terminals,
        failure_terminals=failure_terminals,
        reward_components=reward_components,
    )

    next_state = next_state.replace(terminal_hold_counts=next_terminal_hold_counts)
    if return_next_states:
        return next_state, step_output, next_states
    return next_state, step_output


def vecenv_step_freeflyer(
    state: FreeFlyerVecEnvState,
    commanded_ctrl: jnp.ndarray,
    next_waypoint: jnp.ndarray,
    config: VecEnvStepConfig,
    prev_residuals: Optional[jnp.ndarray] = None,
) -> Tuple[FreeFlyerVecEnvState, VecEnvStepOutput]:
    commanded_ctrl = jnp.asarray(commanded_ctrl)

    prev_states = _compute_freeflyer_state_features(state, next_waypoint)
    prev_obs = _compute_freeflyer_observations(state)
    if config.effects_enabled:
        prepared_state, applied_ctrl, qfrc_applied = _prepare_freeflyer_step(
            state, commanded_ctrl
        )
    else:
        prepared_state = state
        applied_ctrl = commanded_ctrl
        qfrc_applied = jnp.zeros((state.qpos.shape[0], 6), dtype=commanded_ctrl.dtype)

    next_state = jax.lax.fori_loop(
        0,
        config.control_decimation,
        lambda _i, sub_state: _freeflyer_step_dynamics(
            sub_state,
            applied_ctrl,
            qfrc_applied,
            config,
        ),
        prepared_state,
    )
    next_states = _compute_freeflyer_state_features(next_state, next_waypoint)
    next_obs = _compute_freeflyer_observations(next_state)

    pos_curr, vel_curr, att_curr, ang_curr = _reward_components(prev_states, config)
    pos_next, vel_next, att_next, ang_next = _reward_components(next_states, config)
    phi_curr = pos_curr + vel_curr + att_curr + ang_curr
    phi_next = pos_next + vel_next + att_next + ang_next

    success_terminals, next_terminal_hold_counts = _compute_terminals(
        next_states,
        prepared_state.terminal_hold_counts,
        config,
    )
    failure_terminals = _compute_failures(next_states, config)
    terminals = jnp.logical_or(success_terminals, failure_terminals)

    if config.use_adaptive_approach and config.res_dim > 0:
        if prev_residuals is None or prev_residuals.shape[-1] == 0:
            prev_residuals = (
                jnp.atleast_2d(state.actuator_force) @ config.thruster_mixer_T
                - jnp.atleast_2d(state.ctrl) @ config.thruster_mixer_T
            )
        else:
            prev_residuals = jnp.asarray(prev_residuals, dtype=prev_states.dtype)
    else:
        prev_residuals = None

    penalties, penalty_components = _compute_penalties(
        prev_states,
        next_states,
        success_terminals,
        commanded_ctrl,
        prev_residuals,
        config,
    )
    rewards = phi_next - phi_curr - penalties
    terminal_bonus = jnp.asarray(config.terminal_bonus, dtype=rewards.dtype)
    rewards = rewards + terminal_bonus * success_terminals.astype(rewards.dtype)

    if config.collect_reward_components:
        shaping_deltas = jnp.stack(
            [
                pos_next - pos_curr,
                vel_next - vel_curr,
                att_next - att_curr,
                ang_next - ang_curr,
            ],
            axis=1,
        )
        total_shaping = shaping_deltas.sum(axis=1)
        reward_components = {
            "shaping_pos": shaping_deltas[:, 0],
            "shaping_vel": shaping_deltas[:, 1],
            "shaping_att": shaping_deltas[:, 2],
            "shaping_angvel": shaping_deltas[:, 3],
            "shaping_total": total_shaping,
            **penalty_components,
            "bonus_terminal": terminal_bonus
            * success_terminals.astype(rewards.dtype),
            "terminated_success": success_terminals.astype(rewards.dtype),
            "terminated_failure": failure_terminals.astype(rewards.dtype),
            "reward_total": rewards,
        }
    else:
        reward_components = {"reward_total": rewards}

    actual_wrench = jnp.atleast_2d(applied_ctrl) @ config.thruster_mixer_T
    step_output = VecEnvStepOutput(
        prev_states=prev_states,
        next_states=next_states,
        rewards=rewards,
        terminals=terminals,
        commanded_ctrl=commanded_ctrl,
        applied_ctrl=applied_ctrl,
        actual_wrench=actual_wrench,
        desired_wrench=commanded_ctrl @ config.thruster_mixer_T,
        prev_obs=prev_obs,
        next_obs=next_obs,
        success_terminals=success_terminals,
        failure_terminals=failure_terminals,
        reward_components=reward_components,
    )
    next_state = next_state.replace(terminal_hold_counts=next_terminal_hold_counts)
    return next_state, step_output
