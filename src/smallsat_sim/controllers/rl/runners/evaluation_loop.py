import os
from typing import Any

import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.controllers.rl.runners.rollout import (
    AdaptationRolloutExtra,
    FunctionalRolloutCallbacks,
    make_zero_bootstrap_value,
    prepare_policy_input_with_residuals,
    run_functional_rollout,
    update_history_buffer,
)
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules
from smallsat_sim.utils.helpers_jax import (
    calc_attitude_error,
    calc_extrinsic_error,
    calc_lateral_tracking_error,
)


def evaluate_runner(runner: Any, phase: int = 2) -> None:
    """
    Evaluate the runner policy.

    If ``phase == 1``, evaluate the base policy before training adaptation.
    If ``phase == 2``, evaluate with adaptation residual estimates enabled.
    """
    file_path = os.path.join(runner.ckpt_dir, runner.training_state_file_name)
    if os.path.isfile(file_path):
        restored_state = load_trained_modules(
            runner.ckpt_dir, runner.training_state_file_name
        )
        actor_state = restored_state["actor_model"]
        critic_state = restored_state["critic_model"]
        if isinstance(actor_state, dict):
            nnx.update(runner.agent.actor, actor_state)
        else:
            nnx.update(runner.agent.actor.mu_net, actor_state.mu_net)
        if isinstance(critic_state, dict):
            nnx.update(runner.agent.critic, critic_state)
        else:
            nnx.update(runner.agent.critic.v_net, critic_state.v_net)
        if runner.env.use_adaptive_approach and phase == 2:
            adapt_module_state = load_trained_modules(
                runner.ckpt_dir, runner.adaptation_module_file_name
            )
            am_state = adapt_module_state["am_model"]
            if isinstance(am_state, dict):
                nnx.update(runner.am, am_state)
            else:
                nnx.update(runner.am, am_state)
    else:
        raise Exception("Not all necessary modules have been trained yet.\n")

    if phase not in (1, 2):
        raise ValueError("Phase must be either 1 or 2.")

    subkeys_eval = runner._take_keys(runner.n_evals)
    subkeys_eval = jnp.atleast_2d(subkeys_eval)

    num_envs = runner.env.num_envs
    history_len = runner.env.history_len
    state_action_dim = runner.state_action_dim
    returns = jnp.zeros((num_envs, runner.n_evals), dtype=jnp.float32)

    for eval_idx in range(runner.n_evals):
        print(f"Testing policy: episode {eval_idx + 1}/{runner.n_evals}")
        runner.env.reset()
        runner.env.reset_perturbations()

        if runner.env.train_with_failures and eval_idx >= runner.n_evals // 2:
            eval_key = subkeys_eval[eval_idx]
            runner.env.apply_random_perturbations(
                key=eval_key,
                fraction_perturbed_envs=0.4,
                perturbation_distribution=jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
            )
            runner.env.apply_random_disturbance(
                key=eval_key,
                fraction_disturbed_envs=0.1,
            )

        step_config = runner.env.build_step_config(
            max_episode_len=runner.episode_len,
        )
        vec_state = runner.env.state_struct

        if runner.env.use_adaptive_approach:
            residual_init = jnp.zeros((num_envs, runner.env.res_dim), dtype=jnp.float32)
        else:
            residual_init = jnp.zeros((num_envs, 0), dtype=jnp.float32)

        extra = AdaptationRolloutExtra(
            history=jnp.zeros((num_envs, history_len, state_action_dim)),
            counts=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

        def _prepare_policy_input(_step, states, residuals, carry_extra):
            return prepare_policy_input_with_residuals(
                _step, states, residuals, carry_extra
            )

        def _sample_policy(_step, policy_input, rng_key, carry_extra):
            del _step
            actions = runner.agent.get_control_input("evaluation", policy_input)
            zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
            return actions, zeros, zeros, rng_key, carry_extra

        def _post_step(_step, step_output, actions, residuals, reset_flag, carry_extra):
            del _step, residuals
            history, counts, history_full, new_extra = update_history_buffer(
                carry_extra=carry_extra,
                prev_states=step_output.prev_states,
                actions=actions,
                reset_flag=reset_flag,
                history_len=history_len,
            )

            if runner.env.use_adaptive_approach:
                desired = step_output.desired_wrench
                if phase == 1:
                    extrinsic = step_output.actual_wrench
                else:
                    extrinsic = runner.adaptation_module(history)
                residuals_next = extrinsic - desired
            else:
                residuals_next = jnp.zeros(
                    (num_envs, 0), dtype=step_output.actual_wrench.dtype
                )
            return residuals_next, history_full, new_extra

        _bootstrap_value = make_zero_bootstrap_value(num_envs)

        rollout_result = run_functional_rollout(
            step_config=step_config,
            initial_state=vec_state,
            initial_residuals=residual_init,
            rng=runner.agent.key,
            num_steps=runner.episode_len,
            reference_waypoint=runner.reference_point,
            callbacks=FunctionalRolloutCallbacks(
                prepare_policy_input=_prepare_policy_input,
                sample_policy=_sample_policy,
                post_step=_post_step,
                bootstrap_value=_bootstrap_value,
            ),
            extra=extra,
        )

        jax.block_until_ready(rollout_result.actions)
        runner.agent.key = rollout_result.final_rng

        runner.env._state = rollout_result.final_state
        runner.env._rng = rollout_result.final_state.rng
        runner.env.mjx_batch = rollout_result.final_state.mjx_batch
        runner.env.disturbance_states = rollout_result.final_state.disturbance_states
        runner.env.perturbation_states = rollout_result.final_state.perturbation_states
        if hasattr(runner.env, "disturbances") and runner.env.disturbances is not None:
            for obj, snapshot in zip(
                runner.env.disturbances.disturbances,
                rollout_result.final_state.disturbance_states,
                strict=True,
            ):
                obj.state = snapshot
        if hasattr(runner.env, "perturbations") and runner.env.perturbations is not None:
            for obj, snapshot in zip(
                runner.env.perturbations.perturbations,
                rollout_result.final_state.perturbation_states,
                strict=True,
            ):
                obj.state = snapshot
        runner.env._refresh_effect_states()

        step_outputs = rollout_result.step_outputs
        rewards = step_outputs.rewards
        done_masks = rollout_result.done_masks
        terminated_masks = rollout_result.terminated_masks
        residuals_traj = rollout_result.residuals

        done_cum = jnp.cumsum(done_masks.astype(jnp.int32), axis=0)
        before_first_done = done_cum == 0
        first_done_step = jnp.logical_and(done_masks, done_cum == 1)
        first_episode_mask = jnp.logical_or(before_first_done, first_done_step)
        first_episode_mask_f = first_episode_mask.astype(rewards.dtype)

        returns_eval = jnp.sum(rewards * first_episode_mask_f, axis=0)
        returns = returns.at[:, eval_idx].set(returns_eval)

        mask_f = first_episode_mask_f
        denom = jnp.maximum(mask_f.sum(), 1.0)

        obs_seq = step_outputs.next_obs
        flat_obs = obs_seq.reshape(-1, obs_seq.shape[-1])
        tracking_seq = calc_lateral_tracking_error(flat_obs, runner.planner).reshape(
            runner.episode_len, num_envs
        )
        angle_seq = jnp.degrees(calc_attitude_error(flat_obs)).reshape(
            runner.episode_len, num_envs
        )
        tracking_mean = float((tracking_seq * mask_f).sum() / denom)
        angle_mean = float((angle_seq * mask_f).sum() / denom)

        if runner.env.use_adaptive_approach:
            actual_seq = step_outputs.actual_wrench
            if phase == 1:
                extr_seq = actual_seq
            else:
                extr_seq = residuals_traj + step_outputs.desired_wrench
            extrinsic_seq = calc_extrinsic_error(extr_seq, actual_seq)
            mean_extrinsic_error = float((extrinsic_seq * mask_f).sum() / denom)
        else:
            mean_extrinsic_error = 0.0

        terminals_any_eval = jnp.any(
            jnp.logical_and(step_outputs.terminals, first_episode_mask), axis=0
        )
        success_env_count = float(terminals_any_eval.astype(jnp.float32).sum())
        success_rate = float(terminals_any_eval.astype(jnp.float32).mean())
        if runner.env.collect_reward_components:
            terminated_success_eval = step_outputs.reward_components.get(
                "terminated_success"
            )
            terminated_failure_eval = step_outputs.reward_components.get(
                "terminated_failure"
            )
            if terminated_success_eval is not None:
                success_termination_step_count = float(
                    (terminated_success_eval * first_episode_mask_f).sum()
                )
                success_termination_env_rate = float(
                    jnp.any(
                        jnp.logical_and(
                            terminated_success_eval > 0.0, first_episode_mask
                        ),
                        axis=0,
                    )
                    .astype(jnp.float32)
                    .mean()
                )
            else:
                success_termination_step_count = 0.0
                success_termination_env_rate = 0.0

            if terminated_failure_eval is not None:
                failure_termination_step_count = float(
                    (terminated_failure_eval * first_episode_mask_f).sum()
                )
                failure_termination_env_rate = float(
                    jnp.any(
                        jnp.logical_and(
                            terminated_failure_eval > 0.0, first_episode_mask
                        ),
                        axis=0,
                    )
                    .astype(jnp.float32)
                    .mean()
                )
            else:
                failure_termination_step_count = 0.0
                failure_termination_env_rate = 0.0
        else:
            success_termination_step_count = 0.0
            success_termination_env_rate = 0.0
            failure_termination_step_count = 0.0
            failure_termination_env_rate = 0.0
        terminated_step_count = float(terminated_masks.astype(jnp.float32).sum())

        done_count_by_env = first_episode_mask.astype(jnp.int32).sum(axis=0)
        last_active_idx = jnp.maximum(done_count_by_env - 1, 0)
        env_ids = jnp.arange(num_envs, dtype=jnp.int32)
        final_positions = step_outputs.next_obs[last_active_idx, env_ids, :3]
        terminal_envs_at_end = float(
            step_outputs.terminals[last_active_idx, env_ids].astype(jnp.float32).sum()
        )
        terminal_env_rate_at_end = float(
            step_outputs.terminals[last_active_idx, env_ids].astype(jnp.float32).mean()
        )
        ref_pos = jnp.atleast_2d(runner.reference_point)[:, :3]
        final_pos_error = float(jnp.linalg.norm(final_positions - ref_pos, axis=1).mean())

        if runner.agent.has_logger:
            runner.env.logger.log(
                runner.env.run_id,
                float(runner.env.mjx_batch.time[0]),
                step=int(eval_idx),
                run_name=runner.env.run_name,
                stage="evaluation",
                mean_episodic_returns=float(returns_eval.mean()),
                eval_mean_episodic_returns=float(returns_eval.mean()),
                success_env_count=success_env_count,
                success_rate=success_rate,
                terminated_step_count=terminated_step_count,
                success_termination_step_count=success_termination_step_count,
                failure_termination_step_count=failure_termination_step_count,
                success_termination_env_rate=success_termination_env_rate,
                failure_termination_env_rate=failure_termination_env_rate,
                terminal_envs_at_end=terminal_envs_at_end,
                terminal_env_rate_at_end=terminal_env_rate_at_end,
                mean_lateral_error=tracking_mean,
                mean_angle_error=angle_mean,
                mean_extrinsic_error=mean_extrinsic_error,
                mean_final_position_error=final_pos_error,
            )

    print(
        f"Average episodic return over all evals and all envs: {float(returns.mean())}\n"
    )
