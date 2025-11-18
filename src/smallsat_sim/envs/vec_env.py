from argparse import Namespace
from dataclasses import dataclass, replace
from typing import Optional, Tuple, Dict
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.disturbances import (
    DisturbanceStatus,
    DisturbanceState,
    constant_force_apply_from_state,
    disturbance_state_from_serializable,
    disturbance_state_to_serializable,
)
from smallsat_sim.envs.perturbations_rl import (
    PerturbationState,
    PerturbationStatus,
    gp_apply_from_state,
    perturbation_state_from_serializable,
    perturbation_state_to_serializable,
    stuck_off_apply_from_state,
    stuck_on_apply_from_state,
)
from smallsat_sim.utils import xml_parser_lightweight

# JIT-friendly MuJoCo step that advances each environment in parallel.
VMAP_MJX_STEP = jax.vmap(mjx.step, in_axes=(None, 0))


@dataclass
class VecEnvState:
    """
    Minimal container for the mutable pieces of the vectorised environment.

    Storing the RNG key alongside the MJX batch makes it easier to convert the
    environment into a pure functional representation in later steps.
    """

    rng: jnp.ndarray
    mjx_batch: mjx.Data
    disturbance_states: Tuple[Optional[DisturbanceState], ...] = ()
    perturbation_states: Tuple[Optional[PerturbationState], ...] = ()

    def replace(self, **updates) -> "VecEnvState":
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
    lam_wrench_residual: float
    wrench_residual_tolerance: float
    wrench_residual_clip: float
    max_episode_len: int
    res_dim: int
    use_adaptive_approach: bool
    collect_reward_components: bool
    thruster_mixer_T: jnp.ndarray
    base_disturbance_states: Tuple[Optional[DisturbanceState], ...]
    base_perturbation_states: Tuple[Optional[PerturbationState], ...]


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
    reward_components: Dict[str, jnp.ndarray]


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
        reward_components=reward_components,
    )


jax.tree_util.register_pytree_node(
    VecEnvStepOutput,
    _vecenv_step_output_flatten,
    _vecenv_step_output_unflatten,
)


def _vecenv_state_flatten(state: "VecEnvState"):
    children = (
        state.rng,
        state.mjx_batch,
        state.disturbance_states,
        state.perturbation_states,
    )
    return children, None


def _vecenv_state_unflatten(aux_data, children):
    rng, mjx_batch, disturbance_states, perturbation_states = children
    return VecEnvState(
        rng=rng,
        mjx_batch=mjx_batch,
        disturbance_states=tuple(disturbance_states),
        perturbation_states=tuple(perturbation_states),
    )


jax.tree_util.register_pytree_node(
    VecEnvState,
    _vecenv_state_flatten,
    _vecenv_state_unflatten,
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
        disturbance_states=disturbance_states,
        perturbation_states=perturbation_states,
    )


def _sample_random_quat(rng: jnp.ndarray) -> jnp.ndarray:
    """
    Helper to sample a random unit quaternion (uniform over SO(3)).
    """
    key, subkey1, subkey2 = jax.random.split(rng, 3)
    theta1 = jax.random.uniform(key, (1,)) * 2 * jnp.pi
    theta2 = jax.random.uniform(subkey1, (1,)) * 2 * jnp.pi
    theta3 = jax.random.uniform(subkey2, (1,)) * 2 * jnp.pi

    w = jnp.sin(theta1) * jnp.cos(theta2) * jnp.cos(theta3) + jnp.cos(theta1) * jnp.sin(
        theta2
    ) * jnp.sin(theta3)
    x = jnp.cos(theta1) * jnp.sin(theta2) * jnp.cos(theta3) - jnp.sin(theta1) * jnp.cos(
        theta2
    ) * jnp.sin(theta3)
    y = jnp.sin(theta1) * jnp.cos(theta2) * jnp.cos(theta3) - jnp.cos(theta1) * jnp.sin(
        theta2
    ) * jnp.sin(theta3)
    z = jnp.cos(theta1) * jnp.cos(theta2) * jnp.sin(theta3) + jnp.sin(theta1) * jnp.sin(
        theta2
    ) * jnp.cos(theta3)

    quat = jnp.array([w, x, y, z]).reshape(-1)
    return quat / jnp.linalg.norm(quat)


def _quat_log_error(
    q: jnp.ndarray, q_des: jnp.ndarray, eps: float = 1e-9
) -> jnp.ndarray:
    """
    SO(3) log-map (rotation vector) that rotates q -> q_des.
    """

    def _unit(a):
        return a / (jnp.linalg.norm(a) + eps)

    q = _unit(q)
    q_des = _unit(q_des)

    w, x, y, z = q
    qc = jnp.array([w, -x, -y, -z])
    w2, x2, y2, z2 = q_des
    we = w2 * qc[0] - x2 * qc[1] - y2 * qc[2] - z2 * qc[3]
    ex = w2 * qc[1] + x2 * qc[0] + y2 * qc[3] - z2 * qc[2]
    ey = w2 * qc[2] - x2 * qc[3] + y2 * qc[0] + z2 * qc[1]
    ez = w2 * qc[3] + x2 * qc[2] - y2 * qc[1] + z2 * qc[0]
    e = jnp.stack([ex, ey, ez])

    sign = jnp.where(we < 0.0, -1.0, 1.0)
    we = sign * we
    e = sign * e

    e_norm = jnp.linalg.norm(e)
    we_abs = jnp.clip(jnp.abs(we), 0.0, 1.0)
    theta = 2.0 * jnp.arctan2(e_norm, we_abs)
    axis = e / (e_norm + eps)
    logvec = axis * theta

    return logvec


def _compute_state_features(
    mjx_batch: mjx.Data,
    next_waypoint: jnp.ndarray,
) -> jnp.ndarray:
    """
    Pure equivalent of VecEnv.get_states.
    """
    num_envs = mjx_batch.qpos.shape[0]
    reference = jnp.asarray(next_waypoint)
    if reference.ndim == 1:
        reference = jnp.broadcast_to(reference, (num_envs, reference.shape[0]))

    ref_pos = reference[:, :3]
    ref_quat = reference[:, 3:7]

    delta_pos = mjx_batch.qpos[:, 0:3] - ref_pos

    R = mjx_batch.xmat[:, 1, :, :]
    R = jnp.transpose(R, (0, 2, 1))
    vel_body = jnp.einsum("bij,bj->bi", R, mjx_batch.qvel[:, :3])

    attitude_error = jax.vmap(_quat_log_error)(mjx_batch.qpos[:, 3:7], ref_quat)

    states = jnp.concatenate(
        (
            delta_pos,
            attitude_error,
            vel_body,
            mjx_batch.qvel[:, 3:6],
        ),
        axis=1,
    )
    return states


def _compute_observations_from_batch(mjx_batch: mjx.Data) -> jnp.ndarray:
    """
    Pure equivalent of VecEnv.get_obs for a provided MJX batch.
    """
    R = mjx_batch.xmat[:, 1, :, :]
    R = jnp.transpose(R, (0, 2, 1))

    vel_world = mjx_batch.qvel[:, :3]

    def _body_velocity(rot, vel):
        return rot @ vel

    vel_body = jax.vmap(_body_velocity)(R, vel_world)
    obs = jnp.concatenate(
        (
            mjx_batch.qpos,
            vel_body,
            mjx_batch.qvel[:, 3:],
        ),
        axis=1,
    )
    return obs


def _reward_components(
    states: jnp.ndarray,
    config: VecEnvStepConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    dtype = states.dtype
    sigma_pos = jnp.asarray(config.sigma_pos, dtype=dtype)
    sigma_vel = jnp.asarray(config.sigma_vel, dtype=dtype)
    sigma_att = jnp.asarray(config.sigma_att, dtype=dtype)
    sigma_ang = jnp.asarray(config.sigma_angvel, dtype=dtype)
    w_pos = jnp.asarray(config.w_pos, dtype=dtype)
    w_vel = jnp.asarray(config.w_vel, dtype=dtype)
    w_att = jnp.asarray(config.w_att, dtype=dtype)
    w_ang = jnp.asarray(config.w_angvel, dtype=dtype)

    pos_term = jnp.exp(-jnp.sum((states[:, 0:3] / sigma_pos) ** 2, axis=1))
    vel_term = jnp.exp(-jnp.sum((states[:, 6:9] / sigma_vel) ** 2, axis=1))
    att_norm = jnp.linalg.norm(states[:, 3:6], axis=1)
    att_term = jnp.exp(-((att_norm / sigma_att) ** 2))
    ang_term = jnp.exp(-jnp.sum((states[:, 9:12] / sigma_ang) ** 2, axis=1))

    pos_reward = w_pos * pos_term
    vel_reward = w_vel * vel_term
    att_reward = w_att * att_term
    ang_reward = w_ang * ang_term

    return pos_reward, vel_reward, att_reward, ang_reward


def _compute_terminals(states: jnp.ndarray) -> jnp.ndarray:
    radius = jnp.asarray(0.3, dtype=states.dtype)
    return jnp.linalg.norm(states[:, 0:3], axis=1) <= radius


def _compute_penalties(
    prev_states: jnp.ndarray,
    next_states: jnp.ndarray,
    terminals: jnp.ndarray,
    commanded_ctrl: jnp.ndarray,
    prev_residuals: Optional[jnp.ndarray],
    config: VecEnvStepConfig,
) -> Tuple[jnp.ndarray, Dict[str, jnp.ndarray]]:
    """
    Mirror the legacy transition logic:
    - Fuel penalty uses the applied control (same as legacy).
    - Terminal speed penalties depend on the next state's velocity/angular velocity.
    - Wrench residual penalties evaluate the residual slice carried alongside the
      *previous* state (legacy grabbed residuals before the step).
    """
    dtype = prev_states.dtype
    lam_fuel = jnp.asarray(config.lam_fuel, dtype=dtype)
    lam_speed = jnp.asarray(config.lam_speed_terminal, dtype=dtype)
    lam_ang = jnp.asarray(config.lam_ang_speed_terminal, dtype=dtype)
    lam_fuel_term = jnp.asarray(config.lam_fuel_terminal, dtype=dtype)

    fuel_pen = jnp.sum(jnp.abs(commanded_ctrl), axis=1)
    lin_speed_sq = jnp.sum(next_states[:, 6:9] ** 2, axis=1)
    ang_speed_sq = jnp.sum(next_states[:, 9:12] ** 2, axis=1)

    terminal_mask = terminals.astype(dtype)
    vel_pen_terminal = lam_speed * lin_speed_sq * terminal_mask
    angvel_pen_terminal = lam_ang * ang_speed_sq * terminal_mask
    fuel_pen_terminal = lam_fuel_term * fuel_pen * terminal_mask
    fuel_penalty = lam_fuel * fuel_pen

    penalties = (
        fuel_penalty + vel_pen_terminal + angvel_pen_terminal + fuel_pen_terminal
    )

    details = {
        "penalty_fuel": fuel_penalty,
        "penalty_terminal_speed": vel_pen_terminal,
        "penalty_terminal_ang_speed": angvel_pen_terminal,
        "penalty_terminal_fuel": fuel_pen_terminal,
    }

    if config.use_adaptive_approach and config.res_dim > 0:
        lam_residual = jnp.asarray(config.lam_wrench_residual, dtype=dtype)
        tolerance = jnp.asarray(config.wrench_residual_tolerance, dtype=dtype)
        clip_value = jnp.asarray(config.wrench_residual_clip, dtype=dtype)
        if prev_residuals is None or prev_residuals.shape[-1] == 0:
            residuals = jnp.zeros((prev_states.shape[0], config.res_dim), dtype=dtype)
        else:
            residuals = prev_residuals
        residual_norm = jnp.linalg.norm(residuals, axis=1)
        residual_excess = jnp.maximum(residual_norm - tolerance, 0.0)
        residual_clipped = jnp.minimum(residual_excess, clip_value)
        wrench_residual_pen = lam_residual * residual_clipped
        penalties = penalties + wrench_residual_pen
    else:
        wrench_residual_pen = jnp.zeros_like(fuel_penalty)

    details["penalty_wrench_residual"] = wrench_residual_pen
    details["penalty_total"] = penalties
    return penalties, details


def vecenv_reset(
    rng_key: jnp.ndarray,
    *,
    mjx_model: mjx.Model,
    mjx_data_template: mjx.Data,
    mjx_batch_template: mjx.Data,
    init_qpos: jnp.ndarray,
    init_qvel: jnp.ndarray,
    num_envs: int,
    max_start_offset: float,
) -> VecEnvState:
    """
    Functional reset helper mirroring VecEnv.reset.
    """
    rng_key, split_key = jax.random.split(rng_key)
    env_keys = jax.random.split(split_key, num_envs)

    base_data = mjx_data_template.replace(qpos=init_qpos[0], qvel=init_qvel[0])

    def _randomize_state(rng_key):
        pos_key, quat_key = jax.random.split(rng_key)
        random_pos = base_data.qpos[:3] + jax.random.uniform(
            pos_key,
            (3,),
            minval=-max_start_offset,
            maxval=max_start_offset,
        )
        random_quat = _sample_random_quat(quat_key)
        return base_data.replace(qpos=jnp.concatenate([random_pos, random_quat]))

    randomized_batch = jax.vmap(_randomize_state)(env_keys)

    batch = mjx_batch_template.replace(qpos=randomized_batch.qpos, qvel=init_qvel)
    batch = jax.vmap(mjx.forward, in_axes=(None, 0))(mjx_model, batch)

    return VecEnvState(rng=rng_key, mjx_batch=batch)


def prepare_step_functional(
    state: VecEnvState,
    base_ctrl: jnp.ndarray,
) -> Tuple[VecEnvState, jnp.ndarray, jnp.ndarray]:
    """
    Pure helper that applies disturbances and perturbations to produce the control
    and generalized forces for the next MuJoCo step.
    """
    ctrl = base_ctrl
    force = jnp.zeros_like(state.mjx_batch.qfrc_applied)
    updated_disturbances: list[Optional[DisturbanceState]] = []

    for dist_state in state.disturbance_states:
        if dist_state is None:
            updated_disturbances.append(None)
            continue
        const_force = (
            dist_state.params.get("const_force") if dist_state.params else None
        )
        if const_force is not None:
            force_contrib, new_state = constant_force_apply_from_state(
                dist_state,
                state.mjx_batch.time,
                const_force,
            )
            force = force + force_contrib
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
        sim_time = state.mjx_batch.time

        if failure_value == PerturbationStatus.STUCK_OFF.value:
            ctrl, tmp_state = stuck_off_apply_from_state(new_state, ctrl, sim_time)
            if tmp_state is not None:
                new_state = tmp_state
        elif failure_value == PerturbationStatus.STUCK_ON.value:
            ctrl, tmp_state = stuck_on_apply_from_state(new_state, ctrl, sim_time)
            if tmp_state is not None:
                new_state = tmp_state
        elif failure_value in (
            PerturbationStatus.FAULTY_VALVE.value,
            PerturbationStatus.SATURATED_THRUST.value,
            PerturbationStatus.THRUST_INSTABILITY.value,
        ):
            ctrl, tmp_state = gp_apply_from_state(
                new_state,
                ctrl,
                sim_time,
                failure_value,
            )
            if tmp_state is not None:
                new_state = tmp_state
        else:
            # As a fallback, pass through the stuck_off/on helpers so that any
            # mixed-status mask still receives the appropriate adjustments.
            ctrl, tmp_state = stuck_off_apply_from_state(new_state, ctrl, sim_time)
            if tmp_state is not None:
                new_state = tmp_state
            ctrl, tmp_state = stuck_on_apply_from_state(new_state, ctrl, sim_time)
            if tmp_state is not None:
                new_state = tmp_state

        updated_perturbations.append(new_state)

    updated_state = state.replace(
        disturbance_states=tuple(updated_disturbances),
        perturbation_states=tuple(updated_perturbations),
    )

    return updated_state, ctrl, force


def vecenv_step(
    state: VecEnvState,
    commanded_ctrl: jnp.ndarray,
    next_waypoint: jnp.ndarray,
    config: VecEnvStepConfig,
    prev_residuals: Optional[jnp.ndarray] = None,
) -> Tuple[VecEnvState, VecEnvStepOutput]:
    """
    Functional rollout helper that mirrors VecEnv.transition without touching object
    attributes. Returns the updated VecEnvState together with the per-step transition
    data required for training.
    """
    commanded_ctrl = jnp.asarray(commanded_ctrl)

    prev_states = _compute_state_features(state.mjx_batch, next_waypoint)
    prev_obs = _compute_observations_from_batch(state.mjx_batch)
    prepared_state, applied_ctrl, qfrc_applied = prepare_step_functional(
        state, commanded_ctrl
    )

    def _debug_nan(tag: str, tensor: jnp.ndarray):
        def _print(_):
            jax.debug.print("NaN detected in {tag}", tag=tag)
            return jnp.array(0, dtype=jnp.int32)

        _ = jax.lax.cond(
            jnp.isnan(tensor).any(),
            _print,
            lambda _: jnp.array(0, dtype=jnp.int32),
            operand=None,
        )

    _debug_nan("applied_ctrl", applied_ctrl)
    _debug_nan("qfrc_applied", qfrc_applied)

    mjx_batch = prepared_state.mjx_batch.replace(
        ctrl=applied_ctrl,
        qfrc_applied=qfrc_applied,
    )

    def _step_body(_, batch):
        return VMAP_MJX_STEP(config.mjx_model, batch)

    mjx_batch = jax.lax.fori_loop(
        0,
        config.control_decimation,
        lambda i, b: _step_body(i, b),
        mjx_batch,
    )

    next_state = prepared_state.replace(mjx_batch=mjx_batch)
    next_states = _compute_state_features(mjx_batch, next_waypoint)
    next_obs = _compute_observations_from_batch(mjx_batch)
    _debug_nan("mjx_batch.qpos", mjx_batch.qpos)
    _debug_nan("next_obs", next_obs)

    (
        pos_curr,
        vel_curr,
        att_curr,
        ang_curr,
    ) = _reward_components(prev_states, config)
    phi_curr = pos_curr + vel_curr + att_curr + ang_curr

    (
        pos_next,
        vel_next,
        att_next,
        ang_next,
    ) = _reward_components(next_states, config)
    phi_next = pos_next + vel_next + att_next + ang_next

    terminals = _compute_terminals(next_states)
    if config.use_adaptive_approach and config.res_dim > 0:
        if prev_residuals is None or prev_residuals.shape[-1] == 0:
            mixer_T = jnp.asarray(config.thruster_mixer_T, dtype=prev_states.dtype)
            actuator_force = jnp.asarray(
                state.mjx_batch.actuator_force, dtype=prev_states.dtype
            )
            desired_ctrl_prev = jnp.asarray(
                state.mjx_batch.ctrl, dtype=prev_states.dtype
            )
            actual_prev = jnp.atleast_2d(actuator_force) @ mixer_T
            desired_prev = jnp.atleast_2d(desired_ctrl_prev) @ mixer_T
            prev_residuals = actual_prev - desired_prev
        else:
            prev_residuals = jnp.asarray(prev_residuals, dtype=prev_states.dtype)
    else:
        prev_residuals = None

    penalties, penalty_components = _compute_penalties(
        prev_states,
        next_states,
        terminals,
        commanded_ctrl,
        prev_residuals,
        config,
    )

    rewards = phi_next - phi_curr - penalties

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
            "reward_total": rewards,
        }
    else:
        reward_components = {"reward_total": rewards}

    actual_wrench = jnp.atleast_2d(mjx_batch.actuator_force) @ config.thruster_mixer_T

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
        reward_components=reward_components,
    )

    return next_state, step_output


def vecenv_reset_to_config(
    state: VecEnvState,
    config: VecEnvStepConfig,
) -> VecEnvState:
    """
    Reset the VecEnvState using the configuration templates.
    """
    reset_state = vecenv_reset(
        state.rng,
        mjx_model=config.mjx_model,
        mjx_data_template=config.mjx_data_template,
        mjx_batch_template=config.mjx_batch_template,
        init_qpos=config.init_qpos,
        init_qvel=config.init_qvel,
        num_envs=config.num_envs,
        max_start_offset=config.max_start_offset,
    )
    return reset_state.replace(
        disturbance_states=config.base_disturbance_states,
        perturbation_states=config.base_perturbation_states,
    )


class VecEnv(BaseEnv):
    """
    Vectorized environment for the smallsat.
    """

    def __init__(self, args) -> None:
        # Flag to know whether Weights & Biases should be used
        self.use_wandb = args.wandb

        # Flag to decide whether to enable random failures during training (and evaluation)
        self.train_with_failures = self.env_cfg.control.RL.train_with_failures

        # Flag to decide whether to used the pretrained actor and critic networks
        self.use_pretrained = self.env_cfg.control.RL.use_pretrained

        # Flag to decide whether to use the adaptation module
        self.use_adaptive_approach = self.env_cfg.control.RL.use_adaptive_approach

        # Adaptation module architecture
        self.am_architecture = self.env_cfg.control.RL.am_architecture
        self.history_len = self.env_cfg.control.RL.context_window_len

        super().__init__(args)

        # Run ID for logging
        self.run_id = self.env_cfg.control.RL.rl_run_id

        # Number of environments running in parallel
        self.num_envs = self.env_cfg.control.RL.num_envs

        # Mixer maps thruster commands to body-frame wrench
        mixer = jnp.asarray(self.symbolic_model.mixer, dtype=jnp.float32)
        self._thruster_mixer = jax.device_put(mixer)
        self._thruster_mixer_T = jax.device_put(mixer.T)

        # Flag to know whether VecEnv is being used
        self.using_rl = True

        # Observation and action spaces
        self.obs_dim = 12
        self.act_dim = int(self.model.nu)
        if self.use_adaptive_approach is True:
            self.ext_dim = 6
        else:
            self.ext_dim = 0
        self.res_dim = self.ext_dim

        # Initial position and velocity
        self.init_qpos = self.mjx_batch.qpos
        self.init_qvel = self.mjx_batch.qvel

        # Max. offset from the initial position at the start
        self.max_start_offset = self.env_cfg.Bodies.max_start_offset

        # Load mission tolerances, reward weights, and penalty weights
        self._load_vec_env_hyperparams()

        # Perform a Just In Time compilation of mjx.step() so that it runs efficiently on GPU
        self.jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        self.jit_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))

        # Cache for the reward breakdown after each transition (used for logging)
        self._last_reward_components: dict[str, jnp.ndarray] = {}
        self.collect_reward_components = getattr(args, "wandb", False) or getattr(
            args, "log", False
        )
        self.disturbance_states: Tuple[DisturbanceState, ...] = ()
        self.perturbation_states: Tuple[PerturbationState, ...] = ()
        self.reset()
        self._refresh_effect_states()
        self._state = VecEnvState(
            rng=self._rng,
            mjx_batch=self.mjx_batch,
            disturbance_states=self.disturbance_states,
            perturbation_states=self.perturbation_states,
        )
        self._refresh_effect_states()

    def next_rng_keys(self, count: int = 1) -> jnp.ndarray:
        """
        Draw ``count`` fresh PRNG keys from the environment stream.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        splits = jax.random.split(self._rng, count + 1)
        self._rng = splits[0]
        return splits[1:]

    def reset(self) -> None:
        """
        Reset the agent in all the environment instances, while randomizing the initial position.
        """
        new_state = vecenv_reset(
            self._rng,
            mjx_model=self.mjx_model,
            mjx_data_template=self.mjx_data,
            mjx_batch_template=self.mjx_batch,
            init_qpos=self.init_qpos,
            init_qvel=self.init_qvel,
            num_envs=self.num_envs,
            max_start_offset=self.max_start_offset,
        )
        self._rng = new_state.rng
        self.mjx_batch = new_state.mjx_batch
        self.mjx_data = self.mjx_data.replace(
            qpos=self.mjx_batch.qpos[0], qvel=self.mjx_batch.qvel[0]
        )
        self._refresh_effect_states()
        self._state = new_state.replace(
            disturbance_states=self.disturbance_states,
            perturbation_states=self.perturbation_states,
        )
        self._refresh_effect_states()

    def transition(
        self,
        actions: jnp.ndarray,
        states_res: jnp.ndarray,
        next_waypoint: jnp.ndarray,
        iter: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply input action on the environment. Returns the rewards and whether the terminal state has been reached.
        """

        def _reward_components(st):
            """
            Potential-based shaping terms, weighted by the configured reward weights.
            """
            pos_term = jnp.exp(-jnp.sum((st[:, 0:3] / self.sigma_pos) ** 2, axis=1))
            vel_term = jnp.exp(-jnp.sum((st[:, 6:9] / self.sigma_vel) ** 2, axis=1))
            att_term = jnp.exp(
                -((jnp.linalg.norm(st[:, 3:6], axis=1) / self.sigma_att) ** 2)
            )
            ang_term = jnp.exp(-jnp.sum((st[:, 9:12] / self.sigma_angvel) ** 2, axis=1))

            pos_reward = self.w_pos * pos_term
            vel_reward = self.w_vel * vel_term
            att_reward = self.w_att * att_term
            ang_reward = self.w_angvel * ang_term

            return pos_reward, vel_reward, att_reward, ang_reward

        state_features = states_res[:, :12]
        (
            pos_reward_curr,
            vel_reward_curr,
            att_reward_curr,
            ang_reward_curr,
        ) = _reward_components(state_features)
        phi_s = pos_reward_curr + vel_reward_curr + att_reward_curr + ang_reward_curr

        # Step the environment
        self.step(input=actions)

        # Get new states
        next_states = self.get_states(next_waypoint)

        # Post-step potential
        (
            pos_reward_next,
            vel_reward_next,
            att_reward_next,
            ang_reward_next,
        ) = _reward_components(next_states)
        phi_s_next = (
            pos_reward_next + vel_reward_next + att_reward_next + ang_reward_next
        )

        # Check if the agent is out-of-bounds or has reached the goal
        is_terminal = jax.vmap(self._in_terminal_set)
        terminal = is_terminal(next_states)

        # Penalties
        fuel_pen = jnp.sum(jnp.abs(actions), axis=1)
        lin_speed_sq = jnp.sum(next_states[:, 6:9] ** 2, axis=1)
        ang_speed_sq = jnp.sum(next_states[:, 9:12] ** 2, axis=1)
        vel_pen_terminal = jnp.where(
            terminal, self.lam_speed_terminal * lin_speed_sq, 0.0
        )
        angvel_pen_terminal = jnp.where(
            terminal, self.lam_ang_speed_terminal * ang_speed_sq, 0.0
        )
        fuel_pen_terminal = jnp.where(terminal, self.lam_fuel_terminal * fuel_pen, 0.0)
        fuel_penalty = self.lam_fuel * fuel_pen
        penalties = (
            fuel_penalty + vel_pen_terminal + angvel_pen_terminal + fuel_pen_terminal
        )

        if self.use_adaptive_approach and self.res_dim > 0:
            # Wrench residual penalty
            if states_res.shape[1] >= 12 + self.res_dim:
                residual_slice = states_res[:, 12 : 12 + self.res_dim]
            else:
                residual_slice = jnp.zeros(
                    (self.num_envs, self.res_dim),
                    dtype=state_features.dtype,
                )
            residual_norm = jnp.linalg.norm(residual_slice, axis=1)
            lam_residual = jnp.asarray(
                self.lam_wrench_residual, dtype=residual_norm.dtype
            )
            tolerance = jnp.asarray(
                self.wrench_residual_tolerance, dtype=residual_norm.dtype
            )
            clip_value = jnp.asarray(
                self.wrench_residual_clip, dtype=residual_norm.dtype
            )
            residual_excess = jnp.maximum(residual_norm - tolerance, 0.0)
            residual_clipped = jnp.minimum(residual_excess, clip_value)
            wrench_residual_pen = lam_residual * residual_clipped
            penalties = penalties + wrench_residual_pen
        else:
            wrench_residual_pen = jnp.zeros_like(fuel_penalty)

        # Reward shaping
        rewards = phi_s_next - phi_s - penalties

        if self.collect_reward_components:
            shaping_deltas = jnp.stack(
                [
                    pos_reward_next - pos_reward_curr,
                    vel_reward_next - vel_reward_curr,
                    att_reward_next - att_reward_curr,
                    ang_reward_next - ang_reward_curr,
                ],
                axis=1,
            )
            total_shaping = shaping_deltas.sum(axis=1)
            self._last_reward_components = {
                "shaping_pos": shaping_deltas[:, 0],
                "shaping_vel": shaping_deltas[:, 1],
                "shaping_att": shaping_deltas[:, 2],
                "shaping_angvel": shaping_deltas[:, 3],
                "shaping_total": total_shaping,
                "penalty_fuel": fuel_penalty,
                "penalty_terminal_speed": vel_pen_terminal,
                "penalty_terminal_ang_speed": angvel_pen_terminal,
                "penalty_terminal_fuel": fuel_pen_terminal,
                "penalty_wrench_residual": wrench_residual_pen,
                "penalty_total": penalties,
                "reward_total": rewards,
            }
        else:
            self._last_reward_components = {"reward_total": rewards}

        return rewards, terminal

    def get_last_reward_components(self) -> dict[str, jnp.ndarray]:
        """
        Return the shaped reward and penalty breakdown from the most recent transition.
        """
        return self._last_reward_components

    def _refresh_effect_states(self) -> None:
        """
        Capture snapshots of the current disturbance and perturbation objects.
        These will later feed the functional rollout helpers.
        """
        if hasattr(self, "disturbances") and self.disturbances is not None:
            self.disturbance_states = tuple(
                getattr(dist, "state", None) for dist in self.disturbances.disturbances
            )
        else:
            self.disturbance_states = ()

        if hasattr(self, "perturbations") and self.perturbations is not None:
            self.perturbation_states = tuple(
                getattr(pert, "state", None)
                for pert in self.perturbations.perturbations
            )
        else:
            self.perturbation_states = ()

    def step(self, input) -> None:
        """
        Simulate environments for one timestep.
        """
        # Prepare env for simulation step
        self._pre_physics_step(input)

        # Advance simulation
        for substep in range(self.env_cfg.control.control_decimation):
            # Update viewer
            if (
                substep % self.env_cfg.viewer.viewer_decimation == 0
                and hasattr(self, "viewer")
                and self.viewer is not None
            ):
                mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)
                self._update_viewer()

            self.mjx_batch = self.jit_step(self.mjx_model, self.mjx_batch)

        # Execute post physics steps
        self._post_physics_step()

    def get_obs(self) -> jnp.ndarray:
        """
        Return all states.
        """
        # obs = [r (3),
        #        q (4),
        #        v (3), --> in BODY frame
        #        omega (3)]

        # Retrieve current rotation matrix
        R = self.mjx_batch.xmat[:, 1, :, :]

        # Rotate matrix
        R = jnp.transpose(R, (0, 2, 1))

        @jax.vmap
        def multiply_transpose_velocity(R, vel):
            return jnp.matmul(R, vel)  # Shape (3,)

        # Rotate intertial velocity to body velocity
        vel_body = multiply_transpose_velocity(R, self.mjx_batch.qvel[:, :3])

        # Create array of observations
        obs = jnp.concatenate(
            (self.mjx_batch.qpos, vel_body, self.mjx_batch.qvel[:, 3:]), axis=1
        )

        return obs

    def get_states(self, next_waypoint: jnp.ndarray) -> jnp.ndarray:
        """
        Return state vector relative to the provided reference.
        """
        # Retrieve current rotation matrix
        R = self.mjx_batch.xmat[:, 1, :, :]

        # Rotate matrix
        R = jnp.transpose(R, (0, 2, 1))

        @jax.vmap
        def multiply_transpose_velocity(R, vel):
            return jnp.matmul(R, vel)  # Shape (3,)

        # Rotate inertial velocity to body velocity
        vel_body = multiply_transpose_velocity(R, self.mjx_batch.qvel[:, :3])

        # Ensure reference tensors have correct shape
        reference = jnp.asarray(next_waypoint)
        if reference.ndim == 1:
            reference = jnp.broadcast_to(reference, (self.num_envs, reference.shape[0]))

        # Extract position and quaternion reference
        ref_pos = reference[:, :3]
        ref_quat = reference[:, 3:7]

        delta_pos = self.mjx_batch.qpos[:, 0:3] - ref_pos

        get_error_quat = jax.vmap(self._get_error_quat_logvec, in_axes=(0, 0))
        attitude_error = get_error_quat(self.mjx_batch.qpos[:, 3:7], ref_quat)

        states = jnp.concatenate(
            (
                delta_pos,
                attitude_error,
                vel_body,
                self.mjx_batch.qvel[:, 3:6],
            ),
            axis=1,
        )

        return states

    @property
    def state_struct(self) -> VecEnvState:
        """
        Expose the current environment state as a lightweight dataclass.
        """
        return self._state

    def apply_state_struct(self, state: VecEnvState) -> None:
        """
        Overwrite the imperative environment state with the provided functional snapshot.
        """
        self._state = state
        self._rng = state.rng
        self.mjx_batch = state.mjx_batch
        self.disturbance_states = state.disturbance_states
        self.perturbation_states = state.perturbation_states

        if hasattr(self, "disturbances") and self.disturbances is not None:
            for obj, snapshot in zip(
                self.disturbances.disturbances,
                state.disturbance_states,
                strict=True,
            ):
                obj.state = snapshot
        if hasattr(self, "perturbations") and self.perturbations is not None:
            for obj, snapshot in zip(
                self.perturbations.perturbations,
                state.perturbation_states,
                strict=True,
            ):
                obj.state = snapshot
        self._refresh_effect_states()

    def verify_functional_step(
        self,
        state: VecEnvState,
        commanded_ctrl: jnp.ndarray,
        next_waypoint: jnp.ndarray,
        *,
        step_config: VecEnvStepConfig,
        atol: float = 1e-6,
        rtol: float = 1e-5,
    ) -> None:
        """
        Compare the functional step helper against the imperative ``transition`` path
        for a single step, raising ``AssertionError`` when a mismatch is detected.
        """
        if step_config is None:
            raise ValueError("step_config must be provided for verification.")

        commanded_ctrl = jnp.asarray(commanded_ctrl)
        next_waypoint = jnp.asarray(next_waypoint)

        func_state, func_output = vecenv_step(
            state,
            commanded_ctrl,
            next_waypoint,
            step_config,
            prev_residuals=None,
        )

        original_state = self.state_struct

        def _assert_close(name: str, lhs: jnp.ndarray, rhs: jnp.ndarray) -> None:
            if not jnp.allclose(lhs, rhs, atol=atol, rtol=rtol):
                diff = float(jnp.max(jnp.abs(lhs - rhs)))
                raise AssertionError(
                    f"Functional {name} mismatch (max abs diff={diff:.3e})"
                )

        try:
            self.apply_state_struct(state)
            prev_states = _compute_state_features(state.mjx_batch, next_waypoint)

            actual_wrench = self.get_actual_wrench()
            desired_wrench = self.get_desired_wrench(commanded_ctrl)

            res = actual_wrench - desired_wrench

            rewards_legacy, terminals_legacy = self.transition(
                commanded_ctrl,
                jnp.concatenate([prev_states, res], axis=1),
                next_waypoint,
            )
            self._refresh_effect_states()
            legacy_state = VecEnvState(
                rng=self._rng,
                mjx_batch=self.mjx_batch,
                disturbance_states=self.disturbance_states,
                perturbation_states=self.perturbation_states,
            )

            _assert_close("prev_states", func_output.prev_states, prev_states)
            _assert_close(
                "next_states", func_output.next_states, self.get_states(next_waypoint)
            )
            _assert_close("rewards", func_output.rewards, rewards_legacy)
            _assert_close(
                "terminals",
                func_output.terminals.astype(jnp.float32),
                terminals_legacy.astype(jnp.float32),
            )
            _assert_close("applied_ctrl", func_output.applied_ctrl, self.mjx_batch.ctrl)
            _assert_close("actual_wrench", func_output.actual_wrench, actual_wrench)
            _assert_close("desired_wrench", func_output.desired_wrench, desired_wrench)
            _assert_close(
                "mjx_batch.qpos", func_state.mjx_batch.qpos, legacy_state.mjx_batch.qpos
            )
            _assert_close(
                "mjx_batch.qvel", func_state.mjx_batch.qvel, legacy_state.mjx_batch.qvel
            )
            _assert_close(
                "mjx_batch.qfrc_applied",
                func_state.mjx_batch.qfrc_applied,
                legacy_state.mjx_batch.qfrc_applied,
            )
            _assert_close("rng", func_state.rng, legacy_state.rng)

            legacy_components = self.get_last_reward_components()
            for key, value in func_output.reward_components.items():
                if key in legacy_components:
                    _assert_close(
                        f"reward_component[{key}]",
                        value,
                        legacy_components[key],
                    )
        finally:
            self.apply_state_struct(original_state)

    def build_step_config(self, *, max_episode_len: int) -> VecEnvStepConfig:
        """
        Construct the functional step configuration reflecting the current environment.
        """
        return VecEnvStepConfig(
            mjx_model=self.mjx_model,
            mjx_data_template=self.mjx_data,
            mjx_batch_template=self.mjx_batch,
            init_qpos=self.init_qpos,
            init_qvel=self.init_qvel,
            num_envs=self.num_envs,
            max_start_offset=self.max_start_offset,
            control_decimation=int(self.env_cfg.control.control_decimation),
            sigma_pos=float(self.sigma_pos),
            sigma_vel=float(self.sigma_vel),
            sigma_att=float(self.sigma_att),
            sigma_angvel=float(self.sigma_angvel),
            w_pos=float(self.w_pos),
            w_vel=float(self.w_vel),
            w_att=float(self.w_att),
            w_angvel=float(self.w_angvel),
            lam_fuel=float(self.lam_fuel),
            lam_speed_terminal=float(self.lam_speed_terminal),
            lam_ang_speed_terminal=float(self.lam_ang_speed_terminal),
            lam_fuel_terminal=float(self.lam_fuel_terminal),
            lam_wrench_residual=float(self.lam_wrench_residual),
            wrench_residual_tolerance=float(self.wrench_residual_tolerance),
            wrench_residual_clip=float(self.wrench_residual_clip),
            max_episode_len=int(max_episode_len),
            res_dim=int(self.res_dim),
            use_adaptive_approach=bool(self.use_adaptive_approach),
            collect_reward_components=bool(self.collect_reward_components),
            thruster_mixer_T=self._thruster_mixer_T,
            base_disturbance_states=self.disturbance_states,
            base_perturbation_states=self.perturbation_states,
        )

    def apply_random_perturbations(
        self,
        key,
        fraction_perturbed_envs: float,
        perturbation_distribution: jnp.ndarray = jnp.array(
            [0.5, 0.05, 0.15, 0.15, 0.15]
        ),
    ) -> None:
        """
        Apply a perturbation scenario to a subset of the environments (one per environment).
        NOTE: the thrusters are picked at random and the default times are 0.0 for now.
        """
        # Calculate number of environments to perturb
        num_perturbed = int(self.num_envs * fraction_perturbed_envs)
        if num_perturbed == 0:
            return

        # Split the key for permutation and for subkeys for perturbations
        perm_key, subkeys_key = jax.random.split(key, 2)
        subkeys = jax.random.split(subkeys_key, 5)

        # Get a random permutation of all environment indices
        env_indices = jnp.arange(self.num_envs)
        permuted_indices = jax.random.permutation(perm_key, env_indices)
        selected_indices = permuted_indices[:num_perturbed]

        # Determine number of environments for each perturbation based on distribution
        total_weight = jnp.sum(perturbation_distribution)  # should be 1.0
        base_counts = jnp.floor(
            num_perturbed * perturbation_distribution / total_weight
        ).astype(jnp.int32)
        count_sum = int(jnp.sum(base_counts))
        remainder = num_perturbed - count_sum

        # Compute fractional parts for extra allocation
        fractional_parts = (
            num_perturbed * perturbation_distribution / total_weight
        ) - base_counts
        sorted_indices = jnp.argsort(-fractional_parts)  # indices in descending order
        if remainder > 0:
            base_counts = base_counts.at[sorted_indices[:remainder]].add(1)

        # Partition the selected indices by counts into index arrays
        cumulative = 0
        indices_list = []
        for count in base_counts.tolist():
            indices_for_perturbation = selected_indices[cumulative : cumulative + count]
            cumulative += count
            indices_list.append(indices_for_perturbation)

        (
            stuck_off_thruster_envs,
            stuck_on_thruster_envs,
            faulty_valve_envs,
            saturated_thrust_envs,
            thrust_instability_envs,
        ) = indices_list

        # Apply the perturbations with respective subkeys
        self.perturbations.perturbations[0].stuck_off_thruster(
            subkeys[0], stuck_off_thruster_envs
        )
        self.perturbations.perturbations[1].stuck_on_thruster(
            subkeys[1], stuck_on_thruster_envs
        )
        self.perturbations.perturbations[2].register_perturbation(
            subkeys[2], faulty_valve_envs
        )
        self.perturbations.perturbations[3].register_perturbation(
            subkeys[3], saturated_thrust_envs
        )
        self.perturbations.perturbations[4].register_perturbation(
            subkeys[4], thrust_instability_envs
        )

        self._refresh_effect_states()
        self._state = self._state.replace(perturbation_states=self.perturbation_states)

    def apply_random_disturbance(self, key, fraction_disturbed_envs: float) -> None:
        """
        Apply the constant force disturbance to a subset of environments.
        """
        if self.disturbances is None:
            return

        clamped_fraction = max(0.0, min(1.0, fraction_disturbed_envs))
        num_disturbed = int(self.num_envs * clamped_fraction)
        if num_disturbed == 0:
            return

        env_indices = jnp.arange(self.num_envs)
        permuted_indices = jax.random.permutation(key, env_indices)
        selected_indices = permuted_indices[:num_disturbed]

        constant_force_disturbance = None
        for disturbance in self.disturbances.disturbances:
            if (
                getattr(disturbance, "failure_type", None)
                == DisturbanceStatus.CONSTANT_FORCE
            ):
                constant_force_disturbance = disturbance
                break

        if constant_force_disturbance is None:
            return

        constant_force_disturbance.const_force_disturbance(selected_indices)
        self._refresh_effect_states()
        self._state = self._state.replace(disturbance_states=self.disturbance_states)

    def get_desired_wrench(self, ctrl: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the net body-frame wrench generated by the commanded thruster forces.
        """
        ctrl = jnp.asarray(ctrl, dtype=self._thruster_mixer_T.dtype)
        ctrl = jnp.atleast_2d(ctrl)
        wrench = ctrl @ self._thruster_mixer_T
        return wrench

    def get_actual_wrench(self) -> jnp.ndarray:
        """
        Return the body-frame wrench computed from the forces actually applied by MuJoCo.
        """
        actuator_force = jnp.asarray(
            self.mjx_batch.actuator_force, dtype=self._thruster_mixer_T.dtype
        )
        actuator_force = jnp.atleast_2d(actuator_force)
        wrench = actuator_force @ self._thruster_mixer_T
        return wrench

    def _create_viewer(self, args) -> None:
        """
        Creates a viewer to visualize simulation
        """
        # Create instance of MuJoCo viewer
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data_vec[0], key_callback=self._key_callback
        )

        # Set default camera options
        self.viewer.cam.distance = 3.0
        self.viewer.cam.trackbodyid = 1  # tracks smallsat
        self.viewer.cam.azimuth = 10.0
        self.viewer.cam.type = 1

    def _create_renderer(self) -> None:
        """
        Creates a renderer to visualize the experiments (to later save them to a video).
        """
        # Create instance of MuJoCo renderer
        self.renderer = mujoco.Renderer(
            self.model,
            width=self.env_cfg.renderer.width,
            height=self.env_cfg.renderer.height,
        )

        # Set up the scene and the default camera options
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 5.0
        self.cam.trackbodyid = 1  # tracks smallsat
        self.cam.azimuth = 10.0
        self.cam.type = 1

        # Save frames to create the video
        self.frames = []

    def _setup_sim(self, args: Namespace):
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Some subclasses may call into BaseEnv before VecEnv.__init__ has assigned num_envs.
        # Fall back to the configured value so batching still works.
        num_envs = getattr(self, "num_envs", self.env_cfg.control.RL.num_envs)
        self.num_envs = num_envs

        # Generate xml using env and model config files
        xml = xml_parser_lightweight.generate_mujoco_xml(self.env_cfg, self.model_cfg)

        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.mjx_model = mjx.put_model(
            self.model
        )  # impl='warp', warp requires a CUDA device & mujoco-mjx[warp]
        self.mjx_data = mjx.put_data(self.model, self.data)  # impl='warp'

        # Check that MJX puts the JAX arrays on GPU (it should do so automatically)
        print("Devices available to JAX: ", jax.devices())
        print("Device used by JAX: ", self.mjx_data.qpos.devices(), "\n")

        # Batch the data and randomize the starting position
        rng = self.next_rng_keys(num_envs)
        self.mjx_batch = jax.vmap(  # The initial position is randomized when the env is reset (at init and after each epoch)
            lambda rng: self.mjx_data.replace(qpos=self.mjx_data.qpos)
        )(
            rng
        )
        self.data_vec = mjx.get_data(self.model, self.mjx_batch)
        mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)

        # Launch the viewer
        if not args.headless:
            self._create_viewer(args)
        else:
            # If sim is run in headless mode, set the update_viewer method
            # to a lambda function which essentially does nothing
            self.viewer = None
            self._update_viewer = lambda *args, **kwargs: None

        # Launch the renderer to create a video
        if args.video:
            self._create_renderer()
        else:
            # Same logic as for the viewer
            self.renderer = None
            self._update_renderer = lambda *args, **kwargs: None

    def _update_renderer(self):
        """
        Updates the renderer.
        """
        self.renderer.update_scene(self.data_vec[0], self.cam)
        sim_img = self.renderer.render().copy()
        self.frames.append(sim_img)

    def _visualize_renderer(
        self, points: list[np.ndarray], color=[1, 0, 0, 2], size=[0.05, 0, 0]
    ) -> None:
        """
        Visualizes reference points in the MuJoCo renderer.
        """
        if (
            self.data.time >= self.env_cfg.renderer.start_recording
            and self.data.time <= self.env_cfg.renderer.end_recording
        ):
            self.renderer.scene.ngeom = 0

            # Update the renderer scene
            mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)
            self.renderer.update_scene(self.data_vec[0], self.cam)

            # Iterate over all points which need to be visualized in renderer
            for point in points:
                self.renderer.scene.ngeom += 1
                x = point[0]
                y = point[1]
                z = point[2]
                mujoco.mjv_initGeom(
                    self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=size,
                    pos=np.array([x, y, z]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(color),
                )

            # Extract image from renderer and append it for post-processing
            sim_img = self.renderer.render().copy()
            self.frames.append(sim_img)

    def _pre_physics_step(self, input: jnp.ndarray) -> None:
        """
        Prepares the environment for the simulation step in MuJoCo.
        """
        # External disturbances (compatibility path; updates state internally)
        if self.disturbances:
            force = self.disturbances.apply(float(self.mjx_batch.time[0]))
            self.mjx_batch = self.mjx_batch.replace(qfrc_applied=force)

        # Perturbations
        if self.perturbations:
            ctrl = self.perturbations.apply(input, float(self.mjx_batch.time[0]))
            self.mjx_batch = self.mjx_batch.replace(ctrl=ctrl)
        else:
            self.mjx_batch = self.mjx_batch.replace(ctrl=input)

    def _in_terminal_set(self, states: jnp.ndarray) -> jnp.ndarray:
        """
        Returns one if in terminal set, zero otherwise.
        """
        return jnp.linalg.norm(states[0:3]) <= self.sigma_pos

    def _get_error_quat_logvec(
        self, q: jnp.ndarray, q_des: jnp.ndarray, eps: float = 1e-9
    ) -> jnp.ndarray:
        """
        Return SO(3) log-map (rotation vector) that rotates q -> q_des.
        """

        # Normalize
        def _unit(a):
            return a / (jnp.linalg.norm(a) + eps)

        q = _unit(q)
        q_des = _unit(q_des)

        # Error quaternion: q_e = q_des ⊗ conj(q)
        w, x, y, z = q
        qc = jnp.array([w, -x, -y, -z])
        w2, x2, y2, z2 = q_des
        we = w2 * qc[0] - x2 * qc[1] - y2 * qc[2] - z2 * qc[3]
        ex = w2 * qc[1] + x2 * qc[0] + y2 * qc[3] - z2 * qc[2]
        ey = w2 * qc[2] - x2 * qc[3] + y2 * qc[0] + z2 * qc[1]
        ez = w2 * qc[3] + x2 * qc[2] - y2 * qc[1] + z2 * qc[0]
        e = jnp.stack([ex, ey, ez])

        # Enforce shortest path / continuity: flip WHOLE quaternion if we < 0
        sign = jnp.where(we < 0.0, -1.0, 1.0)
        we = sign * we
        e = sign * e

        # log-map: axis * theta, with ||logvec|| = theta in [0, pi]
        e_norm = jnp.linalg.norm(e)  # == sin(theta/2)
        we_abs = jnp.clip(jnp.abs(we), 0.0, 1.0)
        theta = 2.0 * jnp.arctan2(e_norm, we_abs)
        axis = e / (e_norm + eps)
        logvec = axis * theta

        return logvec

    def _get_random_quat(self, rng) -> jnp.ndarray:
        """
        Return a random quaternion.
        """
        return _sample_random_quat(rng)

    def _load_vec_env_hyperparams(self) -> None:
        """
        Load VecEnv-specific hyperparams.
        """
        # Mission tolerances
        self.sigma_pos = self.env_cfg.control.RL.sigma_pos
        self.sigma_vel = self.env_cfg.control.RL.sigma_vel
        self.sigma_att = self.env_cfg.control.RL.sigma_att
        self.sigma_angvel = self.env_cfg.control.RL.sigma_angvel

        # Reward weights
        self.w_pos = self.env_cfg.control.RL.w_pos
        self.w_vel = self.env_cfg.control.RL.w_vel
        self.w_att = self.env_cfg.control.RL.w_att
        self.w_angvel = self.env_cfg.control.RL.w_angvel

        # Penalty weights
        self.lam_fuel = self.env_cfg.control.RL.lam_fuel
        self.lam_speed_terminal = self.env_cfg.control.RL.lam_speed_terminal
        self.lam_ang_speed_terminal = self.env_cfg.control.RL.lam_ang_speed_terminal
        self.lam_fuel_terminal = self.env_cfg.control.RL.lam_fuel_terminal
        self.lam_wrench_residual = self.env_cfg.control.RL.lam_wrench_residual
        self.wrench_residual_tolerance = (
            self.env_cfg.control.RL.wrench_residual_tolerance
        )
        self.wrench_residual_clip = self.env_cfg.control.RL.wrench_residual_clip
