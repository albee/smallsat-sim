import os

import jax
import jax.numpy as jnp

from smallsat_sim.controllers.rl.runners.adaptive_context import (
    authority_bin_stats,
    authority_metrics_from_wrench,
    summarize_authority_metrics,
)
from smallsat_sim.controllers.rl.runners.rollout import (
    FunctionalRolloutCallbacks,
    make_zero_bootstrap_value,
    prepare_policy_input_with_residuals,
    residuals_from_wrench_delta,
    run_functional_rollout,
)
from smallsat_sim.envs.vec_env import (
    _compute_freeflyer_state_features,
    freeflyer_reset_masked,
    vecenv_step_freeflyer,
)


def evaluate_policy_checkpoint(
    runner,
    *,
    key: jnp.ndarray,
    fraction_perturbed_envs: float,
    perturbation_distribution: jnp.ndarray,
    disturbance_fraction: float,
    eval_episodes: int,
) -> float:
    """
    Lightweight evaluation used to safeguard nominal performance.

    The runner's training RNG is restored after evaluation, so checkpoint
    selection does not perturb the training rollout sequence.
    """
    num_envs = runner.env.num_envs
    episode_keys = jax.random.split(key, eval_episodes)
    rewards = []
    agent_key_before = runner.agent.key

    for ep_key in episode_keys:
        runner.env.reset()
        runner.env.reset_perturbations()
        if hasattr(runner.env, "reset_disturbances"):
            runner.env.reset_disturbances()

        if fraction_perturbed_envs > 0.0:
            runner.env.apply_random_perturbations(
                key=ep_key,
                fraction_perturbed_envs=float(fraction_perturbed_envs),
                perturbation_distribution=perturbation_distribution,
            )
        if disturbance_fraction > 0.0:
            runner.env.apply_random_disturbance(
                key=ep_key,
                fraction_disturbed_envs=float(disturbance_fraction),
            )

        step_config = runner.env.build_step_config(max_episode_len=runner.episode_len)
        rollout_backend = os.environ.get(
            "SMALLSAT_ROLLOUT_BACKEND",
            getattr(runner.env.env_cfg.control.RL, "rollout_backend", "mjx"),
        )
        if rollout_backend == "freeflyer":
            initial_state = runner.env.freeflyer_state_struct()
            rollout_step_fn = vecenv_step_freeflyer
            rollout_reset_fn = freeflyer_reset_masked
            rollout_state_features_fn = _compute_freeflyer_state_features
        else:
            initial_state = runner.env.state_struct
            rollout_step_fn = None
            rollout_reset_fn = None
            rollout_state_features_fn = None

        if runner.env.use_adaptive_approach:
            residual_init = jnp.zeros((num_envs, runner.env.res_dim), dtype=jnp.float32)
        else:
            residual_init = jnp.zeros((num_envs, 0), dtype=jnp.float32)

        def _prepare_eval_input(_step, states, residuals, carry_extra):
            del carry_extra
            return prepare_policy_input_with_residuals(_step, states, residuals, None)

        def _sample_eval_policy(_step, policy_input, rng_key, carry_extra):
            del _step, carry_extra
            actions = runner.agent.get_control_input("evaluation", policy_input)
            zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
            return actions, zeros, zeros, rng_key, None

        def _post_eval_step(
            _step, step_output, actions, residuals, reset_flag, carry_extra
        ):
            del _step, actions, reset_flag, carry_extra
            residuals_next = residuals_from_wrench_delta(
                step_output=step_output,
                residuals=residuals,
                use_adaptive_approach=runner.env.use_adaptive_approach,
                adaptive_context_mode=runner.env.adaptive_context_mode,
                thruster_mixer_T=runner.env._thruster_mixer_T,
            )
            return residuals_next, None, None

        rollout_result = run_functional_rollout(
            step_config=step_config,
            initial_state=initial_state,
            initial_residuals=residual_init,
            rng=agent_key_before,
            num_steps=runner.episode_len,
            reference_waypoint=runner.reference_point,
            callbacks=FunctionalRolloutCallbacks(
                prepare_policy_input=_prepare_eval_input,
                sample_policy=_sample_eval_policy,
                post_step=_post_eval_step,
                bootstrap_value=make_zero_bootstrap_value(num_envs),
            ),
            step_fn=rollout_step_fn,
            reset_fn=rollout_reset_fn,
            state_features_fn=rollout_state_features_fn,
        )
        jax.block_until_ready(rollout_result.actions)
        done_masks = rollout_result.done_masks
        done_cum = jnp.cumsum(done_masks.astype(jnp.int32), axis=0)
        first_episode_mask = jnp.logical_or(
            done_cum == 0,
            jnp.logical_and(done_masks, done_cum == 1),
        )
        episodic_returns = (
            rollout_result.step_outputs.rewards * first_episode_mask.astype(jnp.float32)
        ).sum(axis=0)
        rewards.append(float(episodic_returns.mean()))

    runner.agent.key = agent_key_before
    return float(jnp.mean(jnp.asarray(rewards))) if rewards else 0.0


def sample_start_time(key, low: float, high: float) -> float:
    if high <= low:
        return low
    return float(jax.random.uniform(key, (), minval=low, maxval=high))


def sample_flat_indices(total: int, max_samples: int) -> jnp.ndarray:
    if total <= max_samples:
        return jnp.arange(total, dtype=jnp.int32)
    return jnp.linspace(0, total - 1, max_samples, dtype=jnp.int32)


def hold_quality_payload(
    state_seq: jnp.ndarray,
    *,
    terminal_radius: float,
    terminal_max_speed: float,
    terminal_max_att_error: float,
    terminal_max_ang_speed: float,
    terminal_hold_steps: int,
    mask: jnp.ndarray | None = None,
    prefix: str = "hold_quality",
) -> dict[str, float]:
    """
    Diagnose whether policies merely reach the setpoint or can hold it.

    `state_seq` is expected to have shape [T, N, 12] with position, attitude
    vector error, linear velocity, and angular velocity in the standard RL state
    layout. The metrics look at entry into the terminal set before applying the
    consecutive-hold requirement used for success termination.
    """
    if state_seq.shape[0] == 0:
        return {}

    return hold_quality_components_payload(
        position_error=jnp.linalg.norm(state_seq[:, :, :3], axis=-1),
        attitude_error=jnp.linalg.norm(state_seq[:, :, 3:6], axis=-1),
        speed=jnp.linalg.norm(state_seq[:, :, 6:9], axis=-1),
        angular_speed=jnp.linalg.norm(state_seq[:, :, 9:12], axis=-1),
        terminal_radius=terminal_radius,
        terminal_max_speed=terminal_max_speed,
        terminal_max_att_error=terminal_max_att_error,
        terminal_max_ang_speed=terminal_max_ang_speed,
        terminal_hold_steps=terminal_hold_steps,
        mask=mask,
        prefix=prefix,
    )


def hold_quality_components_payload(
    *,
    position_error: jnp.ndarray,
    attitude_error: jnp.ndarray,
    speed: jnp.ndarray,
    angular_speed: jnp.ndarray,
    terminal_radius: float,
    terminal_max_speed: float,
    terminal_max_att_error: float,
    terminal_max_ang_speed: float,
    terminal_hold_steps: int,
    mask: jnp.ndarray | None = None,
    prefix: str = "hold_quality",
) -> dict[str, float]:
    """Same diagnostics as `hold_quality_payload`, using precomputed components."""
    pos_error = jnp.asarray(position_error)
    att_error = jnp.asarray(attitude_error)
    speed = jnp.asarray(speed)
    ang_speed = jnp.asarray(angular_speed)
    in_set = jnp.logical_and(
        pos_error <= terminal_radius,
        jnp.logical_and(
            speed <= terminal_max_speed,
            jnp.logical_and(
                att_error <= terminal_max_att_error,
                ang_speed <= terminal_max_ang_speed,
            ),
        ),
    )

    if mask is None:
        mask = jnp.ones_like(in_set, dtype=bool)
    else:
        mask = jnp.asarray(mask, dtype=bool)
    mask_f = mask.astype(jnp.float32)
    denom = jnp.maximum(mask_f.sum(), 1.0)

    in_set_masked = jnp.logical_and(in_set, mask)
    entered = jnp.any(in_set_masked, axis=0)
    entered_f = entered.astype(jnp.float32)
    entered_count = jnp.maximum(entered_f.sum(), 1.0)
    first_entry = jnp.argmax(in_set_masked.astype(jnp.int32), axis=0)
    time_idx = jnp.arange(state_seq.shape[0], dtype=jnp.int32)[:, None]
    after_entry = jnp.logical_and(
        mask,
        jnp.logical_and(entered[None, :], time_idx >= first_entry[None, :]),
    )
    after_entry_f = after_entry.astype(jnp.float32)
    after_entry_denom = jnp.maximum(after_entry_f.sum(), 1.0)

    post_entry_in_set_fraction = (
        jnp.logical_and(in_set, after_entry).astype(jnp.float32).sum()
        / after_entry_denom
    )
    exited_after_entry = jnp.logical_and(after_entry, jnp.logical_not(in_set))
    exit_after_entry_rate = (
        jnp.any(exited_after_entry, axis=0).astype(jnp.float32) * entered_f
    ).sum() / entered_count
    post_entry_pos_error_mean = (pos_error * after_entry_f).sum() / after_entry_denom
    post_entry_pos_error_max_by_env = jnp.max(
        jnp.where(after_entry, pos_error, 0.0), axis=0
    )
    post_entry_pos_error_max = (
        post_entry_pos_error_max_by_env * entered_f
    ).sum() / entered_count
    post_entry_speed_mean = (speed * after_entry_f).sum() / after_entry_denom
    valid_in_set = jnp.logical_and(in_set, mask)

    def _run_scan(carry, x):
        current, best = carry
        current = jnp.where(x, current + 1, 0)
        best = jnp.maximum(best, current)
        return (current, best), best

    (_, _), best_runs = jax.lax.scan(
        _run_scan,
        (
            jnp.zeros((pos_error.shape[1],), dtype=jnp.int32),
            jnp.zeros((pos_error.shape[1],), dtype=jnp.int32),
        ),
        valid_in_set,
    )
    max_consecutive = best_runs[-1]
    hold_requirement_met = max_consecutive >= int(max(terminal_hold_steps, 1))
    valid_counts = mask.astype(jnp.int32).sum(axis=0)
    last_valid_idx = jnp.maximum(valid_counts - 1, 0)
    env_idx = jnp.arange(pos_error.shape[1], dtype=jnp.int32)
    final_in_set_rate = valid_in_set[last_valid_idx, env_idx].astype(jnp.float32).mean()
    pos_ok = pos_error <= terminal_radius
    speed_ok = speed <= terminal_max_speed
    att_ok = att_error <= terminal_max_att_error
    ang_speed_ok = ang_speed <= terminal_max_ang_speed
    strict_in_set = jnp.logical_and(
        pos_ok,
        jnp.logical_and(speed_ok, jnp.logical_and(att_ok, ang_speed_ok)),
    )
    final_pos_ok = jnp.logical_and(pos_ok[-1], mask[-1]).astype(jnp.float32).mean()
    final_speed_ok = jnp.logical_and(speed_ok[-1], mask[-1]).astype(jnp.float32).mean()
    final_att_ok = jnp.logical_and(att_ok[-1], mask[-1]).astype(jnp.float32).mean()
    final_ang_speed_ok = (
        jnp.logical_and(ang_speed_ok[-1], mask[-1]).astype(jnp.float32).mean()
    )
    final_strict_in_set = (
        jnp.logical_and(strict_in_set[-1], mask[-1]).astype(jnp.float32).mean()
    )

    return {
        f"{prefix}/entered_set_rate": float(entered_f.mean()),
        f"{prefix}/first_entry_step_mean": float(
            (first_entry.astype(jnp.float32) * entered_f).sum() / entered_count
        ),
        f"{prefix}/in_set_fraction": float(
            in_set_masked.astype(jnp.float32).sum() / denom
        ),
        f"{prefix}/post_entry_in_set_fraction": float(post_entry_in_set_fraction),
        f"{prefix}/exit_after_entry_rate": float(exit_after_entry_rate),
        f"{prefix}/post_entry_pos_error_mean": float(post_entry_pos_error_mean),
        f"{prefix}/post_entry_pos_error_max": float(post_entry_pos_error_max),
        f"{prefix}/post_entry_speed_mean": float(post_entry_speed_mean),
        f"{prefix}/max_consecutive_in_set_mean": float(
            max_consecutive.astype(jnp.float32).mean()
        ),
        f"{prefix}/hold_requirement_met_rate": float(
            hold_requirement_met.astype(jnp.float32).mean()
        ),
        f"{prefix}/final_in_set_rate": float(final_in_set_rate),
        f"{prefix}/final_pos_ok_rate": float(final_pos_ok),
        f"{prefix}/final_speed_ok_rate": float(final_speed_ok),
        f"{prefix}/final_att_ok_rate": float(final_att_ok),
        f"{prefix}/final_ang_speed_ok_rate": float(final_ang_speed_ok),
        f"{prefix}/final_strict_in_set_rate": float(final_strict_in_set),
    }


def rollout_authority_payload(
    runner,
    *,
    step_outputs,
    actions_traj,
    success_by_env,
    max_samples: int,
) -> dict[str, float]:
    if max_samples <= 0:
        return {}
    flat_count = runner.steps_per_epoch * runner.env.num_envs
    sample_idx = sample_flat_indices(flat_count, max_samples)
    flat_commanded = actions_traj.reshape(-1, runner.env.act_dim)
    flat_applied = step_outputs.applied_ctrl.reshape(-1, runner.env.act_dim)
    flat_desired = flat_commanded @ runner.env._thruster_mixer_T
    sampled_metrics = authority_metrics_from_wrench(
        commanded_ctrl=flat_commanded[sample_idx],
        applied_ctrl=flat_applied[sample_idx],
        desired_wrench=flat_desired[sample_idx],
        thruster_mixer_T=runner.env._thruster_mixer_T,
    )
    sampled_envs = sample_idx % runner.env.num_envs
    sampled_success = success_by_env[sampled_envs]
    payload = summarize_authority_metrics(sampled_metrics)
    payload.update(
        authority_bin_stats(
            sampled_metrics["normalized_wrench_feasibility_error"],
            sampled_success,
        )
    )
    return payload
