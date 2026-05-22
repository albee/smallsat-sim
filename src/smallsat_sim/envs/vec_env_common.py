from typing import Dict, Optional, Tuple

import jax
import jax.numpy as jnp
from mujoco import mjx

from smallsat_sim.envs.vec_env_types import VecEnvStepConfig


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


def _quat_to_rot_batch(q: jnp.ndarray) -> jnp.ndarray:
    q = q / (jnp.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return jnp.stack(
        [
            jnp.stack(
                [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                axis=1,
            ),
            jnp.stack(
                [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                axis=1,
            ),
            jnp.stack(
                [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
                axis=1,
            ),
        ],
        axis=1,
    )


def _quat_t_batch(q: jnp.ndarray) -> jnp.ndarray:
    q = q / (jnp.linalg.norm(q, axis=1, keepdims=True) + 1e-9)
    w, x, y, z = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    return 0.5 * jnp.stack(
        [
            jnp.stack([-x, -y, -z], axis=1),
            jnp.stack([w, -z, y], axis=1),
            jnp.stack([z, w, -x], axis=1),
            jnp.stack([-y, x, w], axis=1),
        ],
        axis=1,
    )

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


def _compute_terminals(
    states: jnp.ndarray,
    terminal_hold_counts: jnp.ndarray,
    config: VecEnvStepConfig,
) -> Tuple[jnp.ndarray, jnp.ndarray]:
    radius = jnp.asarray(config.terminal_radius, dtype=states.dtype)
    max_speed = jnp.asarray(config.terminal_max_speed, dtype=states.dtype)
    max_att_error = jnp.asarray(config.terminal_max_att_error, dtype=states.dtype)
    max_ang_speed = jnp.asarray(config.terminal_max_ang_speed, dtype=states.dtype)
    required_steps = max(int(config.terminal_hold_steps), 1)
    pos_ok = jnp.linalg.norm(states[:, 0:3], axis=1) <= radius
    speed_ok = jnp.linalg.norm(states[:, 6:9], axis=1) <= max_speed
    att_ok = jnp.linalg.norm(states[:, 3:6], axis=1) <= max_att_error
    ang_speed_ok = jnp.linalg.norm(states[:, 9:12], axis=1) <= max_ang_speed
    in_terminal_set = jnp.logical_and(
        pos_ok,
        jnp.logical_and(speed_ok, jnp.logical_and(att_ok, ang_speed_ok)),
    )
    counts_next = jnp.where(in_terminal_set, terminal_hold_counts + 1, 0)
    terminals = counts_next >= required_steps
    return terminals, counts_next


def _compute_failures(states: jnp.ndarray, config: VecEnvStepConfig) -> jnp.ndarray:
    if not bool(config.enable_failure_termination):
        return jnp.zeros((states.shape[0],), dtype=bool)

    max_pos = jnp.asarray(config.failure_max_position_error, dtype=states.dtype)
    max_speed = jnp.asarray(config.failure_max_speed, dtype=states.dtype)
    max_att = jnp.asarray(config.failure_max_att_error, dtype=states.dtype)
    max_ang = jnp.asarray(config.failure_max_ang_speed, dtype=states.dtype)

    pos_fail = jnp.linalg.norm(states[:, 0:3], axis=1) > max_pos
    speed_fail = jnp.linalg.norm(states[:, 6:9], axis=1) > max_speed
    att_fail = jnp.linalg.norm(states[:, 3:6], axis=1) > max_att
    ang_fail = jnp.linalg.norm(states[:, 9:12], axis=1) > max_ang
    return jnp.logical_or(
        pos_fail,
        jnp.logical_or(speed_fail, jnp.logical_or(att_fail, ang_fail)),
    )


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
            residuals = prev_residuals[:, :6]
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

