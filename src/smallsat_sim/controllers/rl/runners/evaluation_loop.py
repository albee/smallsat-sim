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
from smallsat_sim.controllers.rl.runners.adaptive_context import (
    authority_bin_stats,
    authority_metrics_from_wrench,
    build_adaptation_query,
    build_adaptive_context,
    summarize_authority_metrics,
)
from smallsat_sim.controllers.rl.runners.failure_scenarios import (
    SPLIT_EVAL_ID,
    apply_sampled_failure_scenario_split,
    build_failure_scenario_table,
)
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules
from smallsat_sim.controllers.rl.runners.training_helpers import hold_quality_payload
from smallsat_sim.envs.vec_env import (
    _compute_freeflyer_state_features,
    freeflyer_to_mjx_state,
    vecenv_step_freeflyer,
    freeflyer_reset_masked,
)
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
    cfg = runner.env.env_cfg.control.RL
    failure_start_time_min = float(
        getattr(cfg, "curriculum_failure_start_time_min", 0.0)
    )
    failure_start_time_max = float(
        getattr(cfg, "curriculum_failure_start_time_max", 0.0)
    )
    disturbance_start_time_min = float(
        getattr(cfg, "curriculum_disturbance_start_time_min", 0.0)
    )
    disturbance_start_time_max = float(
        getattr(cfg, "curriculum_disturbance_start_time_max", 0.0)
    )
    scenario_table = None
    if bool(getattr(cfg, "use_controllable_failure_scenarios", False)):
        scenario_table = build_failure_scenario_table(
            runner.env._thruster_mixer_T,
            runner.agent.actor.act_low,
            runner.agent.actor.act_high,
            max_faults=int(getattr(cfg, "failure_scenario_max_faults", 2)),
            min_rank=int(getattr(cfg, "failure_scenario_min_rank", 6)),
            stress_quantile=float(
                getattr(cfg, "failure_scenario_stress_quantile", 0.9)
            ),
            mild_effectiveness=float(
                getattr(cfg, "failure_scenario_mild_effectiveness", 0.5)
            ),
        )
    authority_logging_max_samples = int(
        getattr(cfg, "authority_logging_max_samples", 8192)
    )
    authority_logging_max_samples = max(0, authority_logging_max_samples)

    def _sample_start_time(key, low: float, high: float) -> float:
        if high <= low:
            return low
        return float(jax.random.uniform(key, (), minval=low, maxval=high))

    def _sample_flat_indices(total: int, max_samples: int) -> jnp.ndarray:
        if total <= max_samples:
            return jnp.arange(total, dtype=jnp.int32)
        return jnp.linspace(0, total - 1, max_samples, dtype=jnp.int32)

    for eval_idx in range(runner.n_evals):
        print(f"Testing policy: episode {eval_idx + 1}/{runner.n_evals}")
        runner.env.reset()
        runner.env.reset_perturbations()
        if hasattr(runner.env, "reset_disturbances"):
            runner.env.reset_disturbances()

        if runner.env.train_with_failures and eval_idx >= runner.n_evals // 2:
            eval_key = subkeys_eval[eval_idx]
            (
                perturb_key,
                disturb_key,
                failure_onset_key,
                disturbance_onset_key,
            ) = jax.random.split(eval_key, 4)
            failure_start_time = _sample_start_time(
                failure_onset_key,
                failure_start_time_min,
                failure_start_time_max,
            )
            disturbance_start_time = _sample_start_time(
                disturbance_onset_key,
                disturbance_start_time_min,
                disturbance_start_time_max,
            )
            if scenario_table is not None:
                apply_sampled_failure_scenario_split(
                    runner.env,
                    key=perturb_key,
                    table=scenario_table,
                    split_id=SPLIT_EVAL_ID,
                    fraction_perturbed_envs=0.4,
                    start_time=failure_start_time,
                )
            else:
                runner.env.apply_random_perturbations(
                    key=perturb_key,
                    fraction_perturbed_envs=0.4,
                    perturbation_distribution=jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                    start_time=failure_start_time,
                )
            runner.env.apply_random_disturbance(
                key=disturb_key,
                fraction_disturbed_envs=0.1,
                start_time=disturbance_start_time,
            )

        step_config = runner.env.build_step_config(
            max_episode_len=runner.episode_len,
        )
        rollout_backend = os.environ.get(
            "SMALLSAT_ROLLOUT_BACKEND",
            getattr(runner.env.env_cfg.control.RL, "rollout_backend", "mjx"),
        )
        if rollout_backend == "freeflyer":
            vec_state = runner.env.freeflyer_state_struct()
            rollout_step_fn = vecenv_step_freeflyer
            rollout_reset_fn = freeflyer_reset_masked
            rollout_state_features_fn = _compute_freeflyer_state_features
        else:
            vec_state = runner.env.state_struct
            rollout_step_fn = None
            rollout_reset_fn = None
            rollout_state_features_fn = None

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
            del _step
            history, counts, history_full, new_extra = update_history_buffer(
                carry_extra=carry_extra,
                prev_states=step_output.prev_states,
                actions=actions,
                reset_flag=reset_flag,
                history_len=history_len,
            )

            if runner.env.use_adaptive_approach:
                if phase == 1:
                    residuals_next = build_adaptive_context(
                        commanded_ctrl=step_output.commanded_ctrl,
                        applied_ctrl=step_output.applied_ctrl,
                        actual_wrench=step_output.actual_wrench,
                        desired_wrench=step_output.desired_wrench,
                        previous_context=residuals,
                        use_adaptive_approach=True,
                        adaptive_context_mode=runner.env.adaptive_context_mode,
                        thruster_mixer_T=runner.env._thruster_mixer_T,
                    )
                else:
                    query = build_adaptation_query(
                        states=step_output.prev_states,
                        desired_wrench=step_output.desired_wrench,
                        use_task_conditioned_am=runner.env.use_task_conditioned_am,
                    )
                    context_pred = runner.adaptation_module(history, query)
                    residuals_next = jnp.where(
                        history_full[:, None],
                        context_pred,
                        residuals,
                    )
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
            step_fn=rollout_step_fn,
            reset_fn=rollout_reset_fn,
            state_features_fn=rollout_state_features_fn,
        )

        jax.block_until_ready(rollout_result.actions)
        runner.agent.key = rollout_result.final_rng

        if rollout_backend == "freeflyer":
            synced_state = freeflyer_to_mjx_state(
                rollout_result.final_state,
                runner.env.state_struct,
                step_config,
            )
        else:
            synced_state = rollout_result.final_state

        runner.env._state = synced_state
        runner.env._rng = synced_state.rng
        runner.env.mjx_batch = synced_state.mjx_batch
        runner.env.disturbance_states = synced_state.disturbance_states
        runner.env.perturbation_states = synced_state.perturbation_states
        if hasattr(runner.env, "disturbances") and runner.env.disturbances is not None:
            for obj, snapshot in zip(
                runner.env.disturbances.disturbances,
                synced_state.disturbance_states,
                strict=True,
            ):
                obj.state = snapshot
        if hasattr(runner.env, "perturbations") and runner.env.perturbations is not None:
            for obj, snapshot in zip(
                runner.env.perturbations.perturbations,
                synced_state.perturbation_states,
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
                extr_seq = residuals_traj[..., :6] + step_outputs.desired_wrench
            extrinsic_seq = calc_extrinsic_error(extr_seq, actual_seq)
            mean_extrinsic_error = float((extrinsic_seq * mask_f).sum() / denom)
        else:
            mean_extrinsic_error = 0.0

        terminals_any_eval = jnp.any(
            jnp.logical_and(step_outputs.success_terminals, first_episode_mask), axis=0
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
        hold_payload = hold_quality_payload(
            step_outputs.next_states,
            terminal_radius=float(runner.env.terminal_radius),
            terminal_max_speed=float(runner.env.terminal_max_speed),
            terminal_max_att_error=float(runner.env.terminal_max_att_error),
            terminal_max_ang_speed=float(runner.env.terminal_max_ang_speed),
            terminal_hold_steps=int(runner.env.terminal_hold_steps),
            mask=first_episode_mask,
        )

        authority_payload = {}
        if runner.agent.has_logger and authority_logging_max_samples > 0:
            flat_count = runner.episode_len * num_envs
            sample_idx = _sample_flat_indices(
                flat_count, authority_logging_max_samples
            )
            authority_metrics = authority_metrics_from_wrench(
                commanded_ctrl=step_outputs.commanded_ctrl.reshape(
                    -1, runner.env.act_dim
                )[sample_idx],
                applied_ctrl=step_outputs.applied_ctrl.reshape(-1, runner.env.act_dim)[
                    sample_idx
                ],
                desired_wrench=step_outputs.desired_wrench.reshape(-1, 6)[sample_idx],
                thruster_mixer_T=runner.env._thruster_mixer_T,
            )
            sampled_envs = sample_idx % num_envs
            authority_payload = summarize_authority_metrics(authority_metrics)
            authority_payload.update(
                authority_bin_stats(
                    authority_metrics["normalized_wrench_feasibility_error"],
                    terminals_any_eval[sampled_envs],
                )
            )

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
                **hold_payload,
                **authority_payload,
            )

    print(
        f"Average episodic return over all evals and all envs: {float(returns.mean())}\n"
    )
