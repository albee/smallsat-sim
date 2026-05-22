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
        initial_state = runner.env.state_struct

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
