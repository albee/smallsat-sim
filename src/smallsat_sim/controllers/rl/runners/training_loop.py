import os
import time

import jax
import jax.numpy as jnp
from flax import nnx
import wandb

from smallsat_sim.controllers.rl.runners.rollout import (
    FunctionalRolloutCallbacks,
    make_zero_bootstrap_value,
    prepare_policy_input_with_residuals,
    residuals_from_wrench_delta,
    run_functional_rollout,
)
from smallsat_sim.controllers.rl.runners.curriculum import (
    build_failure_curriculum,
    uniform_failure_distribution,
)
from smallsat_sim.controllers.rl.runners.adaptive_context import (
    authority_bin_stats,
    authority_metrics_from_wrench,
    summarize_authority_metrics,
)
from smallsat_sim.controllers.rl.runners.runner_timing import (
    EpochTiming,
    format_epoch_timing_line,
)
from smallsat_sim.controllers.rl.runners.runner_metrics import (
    build_logger_policy_training_payload,
    build_wandb_policy_training_payload,
)
from smallsat_sim.controllers.rl.runners.runner_utils import (
    load_trained_modules,
    save_training_data,
    save_trained_modules,
)
from smallsat_sim.controllers.rl.storage.replay_buffer import ReplayBuffer
from smallsat_sim.envs.vec_env import _compute_state_features
from smallsat_sim.utils.helpers_jax import (
    calc_attitude_error,
    calc_lateral_tracking_error,
)


def learn_runner(self) -> None:
    """
    Main training loop.
    """
    # Check if training has already been done
    file_path = os.path.join(self.ckpt_dir, self.training_state_file_name)
    if os.path.isfile(file_path):
        return

    if self.env.use_pretrained:
        # Check if pretrained actor and critic modules are available and load them
        file_path = os.path.join(self.ckpt_dir, self.pretraining_state_file_name)
        if os.path.isfile(file_path):
            restored_state = load_trained_modules(
                self.ckpt_dir, self.pretraining_state_file_name
            )
            actor_state = restored_state["actor_model"]
            critic_state = restored_state["critic_model"]
            if isinstance(actor_state, dict):
                nnx.update(self.agent.actor, actor_state)
            else:
                nnx.update(self.agent.actor.mu_net, actor_state.mu_net)
            if isinstance(critic_state, dict):
                nnx.update(self.agent.critic, critic_state)
            else:
                nnx.update(self.agent.critic.v_net, critic_state.v_net)
        else:
            print("No pretrained modules available.\n")

    print("Training agent...\n")

    # Set up buffer
    buffer = ReplayBuffer(
        self.env.num_envs,
        self.env.obs_dim,
        self.env.act_dim,
        self.env.res_dim,
        self.steps_per_epoch,
        self.gamma,
        self.lam,
    )

    # Trigger compilation of jitted updates before the main loop
    self.agent.warmup_jit()

    # Initialize the environment
    self.env.reset()
    self.env.reset_perturbations()
    if hasattr(self.env, "reset_disturbances"):
        self.env.reset_disturbances()
    episode_counter = 0

    # Curriculum knobs live in config so they can be tuned without code edits
    cfg = self.env.env_cfg.control.RL
    nominal_epochs = int(cfg.curriculum_nominal_epochs)
    phase_epochs = int(cfg.curriculum_phase_epochs)
    failure_fraction = float(cfg.curriculum_failure_fraction)
    disturbance_fraction = float(cfg.curriculum_disturbance_fraction)
    critic_warmup_epochs = int(cfg.curriculum_critic_warmup_epochs)
    critic_warmup_scale = float(cfg.curriculum_critic_warmup_scale)
    eval_interval = int(cfg.curriculum_eval_interval)
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
    authority_logging_max_samples = int(
        getattr(cfg, "authority_logging_max_samples", 8192)
    )
    authority_logging_interval = int(getattr(cfg, "authority_logging_interval", 10))
    authority_logging_max_samples = max(0, authority_logging_max_samples)
    authority_logging_interval = max(1, authority_logging_interval)
    checkpoint_interval = int(getattr(cfg, "training_checkpoint_interval", 10))
    checkpoint_interval = max(1, checkpoint_interval)
    # Keep evaluation lightweight relative to training rollouts.
    eval_episodes = max(1, min(self.n_evals, 3))

    # Build phases: nominal -> sequential failures -> disturbances
    phases, total_epochs = build_failure_curriculum(
        train_with_failures=bool(self.env.train_with_failures),
        fallback_epochs=nominal_epochs,
        nominal_epochs=nominal_epochs,
        phase_epochs=phase_epochs,
        failure_fraction=failure_fraction,
        disturbance_fraction=disturbance_fraction,
    )
    # Align runner/agent epoch counts with the curriculum length
    self.epochs = total_epochs
    if hasattr(self.agent, "epochs"):
        self.agent.epochs = total_epochs

    # Track the best checkpoint by nominal score to guard against drift
    best_nominal_score = float("-inf")
    best_actor_state = None
    best_critic_state = None

    def _eval_policy(
        *,
        key: jnp.ndarray,
        fraction_perturbed_envs: float,
        perturbation_distribution: jnp.ndarray,
        disturbance_fraction: float,
    ) -> float:
        """
        Lightweight evaluation used to safeguard nominal performance.
        We keep the training RNG state fixed across evals.
        """
        num_envs = self.env.num_envs
        episode_keys = jax.random.split(key, eval_episodes)
        rewards = []
        # Snapshot the training RNG so evaluation does not consume it.
        agent_key_before = self.agent.key

        for ep_key in episode_keys:
            self.env.reset()
            self.env.reset_perturbations()
            if hasattr(self.env, "reset_disturbances"):
                self.env.reset_disturbances()

            if fraction_perturbed_envs > 0.0:
                self.env.apply_random_perturbations(
                    key=ep_key,
                    fraction_perturbed_envs=float(fraction_perturbed_envs),
                    perturbation_distribution=perturbation_distribution,
                )
            if disturbance_fraction > 0.0:
                self.env.apply_random_disturbance(
                    key=ep_key,
                    fraction_disturbed_envs=float(disturbance_fraction),
                )

            step_config = self.env.build_step_config(
                max_episode_len=self.episode_len
            )
            initial_state = self.env.state_struct

            if self.env.use_adaptive_approach:
                residual_init = jnp.zeros(
                    (num_envs, self.env.res_dim), dtype=jnp.float32
                )
            else:
                residual_init = jnp.zeros((num_envs, 0), dtype=jnp.float32)

            def _prepare_eval_input(_step, states, residuals, carry_extra):
                del carry_extra
                return prepare_policy_input_with_residuals(
                    _step, states, residuals, None
                )

            def _sample_eval_policy(_step, policy_input, rng_key, carry_extra):
                del _step, carry_extra  # unused
                actions = self.agent.get_control_input("evaluation", policy_input)
                zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
                return actions, zeros, zeros, rng_key, None

            def _post_eval_step(
                _step, step_output, actions, residuals, reset_flag, carry_extra
            ):
                del _step, actions, reset_flag, carry_extra  # unused
                residuals_next = residuals_from_wrench_delta(
                    step_output=step_output,
                    residuals=residuals,
                    use_adaptive_approach=self.env.use_adaptive_approach,
                    adaptive_context_mode=self.env.adaptive_context_mode,
                    thruster_mixer_T=self.env._thruster_mixer_T,
                )
                return residuals_next, None, None

            _bootstrap_eval = make_zero_bootstrap_value(num_envs)

            rollout_result = run_functional_rollout(
                step_config=step_config,
                initial_state=initial_state,
                initial_residuals=residual_init,
                rng=agent_key_before,
                num_steps=self.episode_len,
                reference_waypoint=self.reference_point,
                callbacks=FunctionalRolloutCallbacks(
                    prepare_policy_input=_prepare_eval_input,
                    sample_policy=_sample_eval_policy,
                    post_step=_post_eval_step,
                    bootstrap_value=_bootstrap_eval,
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
                rollout_result.step_outputs.rewards
                * first_episode_mask.astype(jnp.float32)
            ).sum(axis=0)
            rewards.append(float(episodic_returns.mean()))

        # Restore the training RNG after evaluation.
        self.agent.key = agent_key_before
        return float(jnp.mean(jnp.asarray(rewards))) if rewards else 0.0

    global_epoch = 0
    # Convenience distribution for nominal-only evaluation.
    zeros_dist = jnp.zeros((5,), dtype=jnp.float32)

    def _sample_start_time(key, low: float, high: float) -> float:
        if high <= low:
            return low
        return float(jax.random.uniform(key, (), minval=low, maxval=high))

    def _sample_flat_indices(total: int, max_samples: int) -> jnp.ndarray:
        if total <= max_samples:
            return jnp.arange(total, dtype=jnp.int32)
        return jnp.linspace(0, total - 1, max_samples, dtype=jnp.int32)

    def _rollout_authority_payload(step_outputs, success_by_env) -> dict[str, float]:
        if authority_logging_max_samples <= 0:
            return {}
        flat_count = self.steps_per_epoch * self.env.num_envs
        sample_idx = _sample_flat_indices(flat_count, authority_logging_max_samples)
        flat_commanded = step_outputs.commanded_ctrl.reshape(-1, self.env.act_dim)
        flat_applied = step_outputs.applied_ctrl.reshape(-1, self.env.act_dim)
        flat_desired = step_outputs.desired_wrench.reshape(-1, 6)
        sampled_metrics = authority_metrics_from_wrench(
            commanded_ctrl=flat_commanded[sample_idx],
            applied_ctrl=flat_applied[sample_idx],
            desired_wrench=flat_desired[sample_idx],
            thruster_mixer_T=self.env._thruster_mixer_T,
        )
        sampled_envs = sample_idx % self.env.num_envs
        sampled_success = success_by_env[sampled_envs]
        payload = summarize_authority_metrics(sampled_metrics)
        payload.update(
            authority_bin_stats(
                sampled_metrics["normalized_wrench_feasibility_error"],
                sampled_success,
            )
        )
        return payload

    for phase_idx, phase in enumerate(phases):
        phase_name = phase["name"]
        phase_epochs = int(phase["epochs"])
        active_failures = list(phase["active_failures"])
        phase_failure_fraction = float(phase["failure_fraction"])
        phase_disturbance_fraction = float(phase["disturbance_fraction"])
        phase_distribution = uniform_failure_distribution(active_failures)
        new_failure_idx = phase["new_failure"]

        print(
            f"[Curriculum] Phase {phase_idx + 1}/{len(phases)}: {phase_name} "
            f"for {phase_epochs} epochs (failure_fraction={phase_failure_fraction:.2f}, "
            f"disturbance_fraction={phase_disturbance_fraction:.2f})"
        )

        for phase_epoch in range(phase_epochs):
            global_epoch += 1
            epoch_start_time = time.perf_counter()
            epoch_key = self._take_keys()
            (
                perturb_key,
                disturb_key,
                actor_key,
                eval_key,
                failure_onset_key,
                disturbance_onset_key,
            ) = jax.random.split(
                epoch_key, 6
            )
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

            # Apply failures/disturbances for this phase with fixed proportions
            if self.env.train_with_failures:
                self.env.reset_perturbations()  # avoid accumulating failures across epochs
                if hasattr(self.env, "reset_disturbances"):
                    self.env.reset_disturbances()
                if phase_failure_fraction > 0.0:
                    self.env.apply_random_perturbations(
                        key=perturb_key,
                        fraction_perturbed_envs=phase_failure_fraction,
                        perturbation_distribution=phase_distribution,
                        start_time=failure_start_time,
                    )
                if phase_disturbance_fraction > 0.0:
                    self.env.apply_random_disturbance(
                        key=disturb_key,
                        fraction_disturbed_envs=phase_disturbance_fraction,
                        start_time=disturbance_start_time,
                    )

            # Accumulate rollout stats to emit once per epoch
            scan_start = time.perf_counter()
            step_config = self.env.build_step_config(
                max_episode_len=self.max_ep_len
            )
            initial_state = self.env.state_struct
            actor_state, critic_state = self.agent.actor_critic_state()

            # Residuals are the adaptation signal; keep shape consistent even when disabled
            if self.env.use_adaptive_approach:
                residual_init = jnp.zeros(
                    (self.env.num_envs, self.env.res_dim), dtype=jnp.float32
                )
            else:
                residual_init = jnp.zeros((self.env.num_envs, 0), dtype=jnp.float32)

            # The rollout helper is fully functional. These callbacks thread
            # policy logic and residual updates into the environment scan
            def _prepare_policy_input(_step, states, residuals, carry_extra):
                return prepare_policy_input_with_residuals(
                    _step, states, residuals, carry_extra
                )

            def _sample_policy(_step, policy_input, rng_key, carry_extra):
                del _step  # unused
                rng_key, sample_key = jax.random.split(rng_key)
                actions, values, logp = self.agent.functional_act(
                    actor_state,
                    critic_state,
                    policy_input,
                    sample_key,
                )
                return actions, values, logp, rng_key, carry_extra

            def _post_step(
                _step, step_output, actions, residuals, reset_flag, carry_extra
            ):
                del _step, actions, reset_flag  # unused
                residuals_next = residuals_from_wrench_delta(
                    step_output=step_output,
                    residuals=residuals,
                    use_adaptive_approach=self.env.use_adaptive_approach,
                    adaptive_context_mode=self.env.adaptive_context_mode,
                    thruster_mixer_T=self.env._thruster_mixer_T,
                )
                return residuals_next, None, carry_extra

            def _bootstrap_value(
                step_idx, env_state, residuals, rng_key, carry_extra
            ):
                rng_key, value_key = jax.random.split(rng_key)
                # Bootstrap with the critic on the next observation
                next_states = _compute_state_features(
                    env_state.mjx_batch, self.reference_point
                )
                policy_input, carry_extra = _prepare_policy_input(
                    step_idx, next_states, residuals, carry_extra
                )
                _, values, _ = self.agent.functional_act(
                    actor_state,
                    critic_state,
                    policy_input,
                    value_key,
                )
                return values, rng_key, carry_extra

            # Run a full epoch rollout in one compiled scan
            rollout_result = run_functional_rollout(
                step_config=step_config,
                initial_state=initial_state,
                initial_residuals=residual_init,
                rng=self.agent.key,
                num_steps=self.steps_per_epoch,
                reference_waypoint=self.reference_point,
                callbacks=FunctionalRolloutCallbacks(
                    prepare_policy_input=_prepare_policy_input,
                    sample_policy=_sample_policy,
                    post_step=_post_step,
                    bootstrap_value=_bootstrap_value,
                ),
            )

            # Materialize device work before timing/logging
            jax.block_until_ready(rollout_result.actions)
            scan_time = time.perf_counter() - scan_start
            self.agent.key = rollout_result.final_rng

            if (
                self._functional_check_enabled
                and not self._functional_check_ran
                and rollout_result.actions.shape[0] > 0
            ):
                self.env.verify_functional_step(
                    initial_state,
                    rollout_result.actions[0],
                    self.reference_point,
                    step_config=step_config,
                    atol=self._functional_check_atol,
                    rtol=self._functional_check_rtol,
                )
                self._functional_check_ran = True

            # Sync the imperative env state with the functional rollout result
            self.env.apply_state_struct(rollout_result.final_state)

            # Unpack rollout tensors for buffer storage and logging
            step_outputs = rollout_result.step_outputs
            actions_traj = rollout_result.actions
            values_traj = rollout_result.values
            logp_traj = rollout_result.logp
            residuals_traj = rollout_result.residuals
            done_masks = rollout_result.done_masks
            terminated_masks = rollout_result.terminated_masks
            truncated_masks = rollout_result.truncated_masks
            bootstrap_vals = rollout_result.bootstrap_values
            episode_returns_traj = rollout_result.episode_returns

            buffer_start = time.perf_counter()
            # Store the full trajectory in the device-friendly replay buffer
            buffer.store_batch(
                step_outputs.prev_states,
                actions_traj,
                step_outputs.rewards,
                values_traj,
                logp_traj,
                residuals_traj,
            )
            buffer.finalize_with_masks(
                done_masks=done_masks,
                bootstrap_values=bootstrap_vals,
                terminated_masks=terminated_masks,
                truncated_masks=truncated_masks,
            )

            done_events = done_masks.astype(jnp.float32)
            episode_return_sum = 0.0
            episode_counter_epoch = 0

            if float(done_events.sum()) > 0.0:
                done_returns = episode_returns_traj * done_events
                episode_return_sum = float(done_returns.sum())
                episode_counter_epoch = int(done_events.sum())

                if self.agent.has_logger:
                    flat_done_returns = jax.device_get(done_returns).reshape(-1)
                    for mean_value in flat_done_returns.tolist():
                        if mean_value == 0.0:
                            continue
                        self.env.logger.log(
                            self.env.run_id,
                            float(self.env.mjx_batch.time[0]),
                            step=episode_counter,
                            run_name=self.env.run_name,
                            stage="policy_training",
                            mean_episodic_returns=mean_value,
                        )
                        episode_counter += 1

            buffer_time = time.perf_counter() - buffer_start

            epoch_reward_components = (
                step_outputs.reward_components
                if self.env.collect_reward_components
                else {}
            )

            # Reset the imperative environment for the next epoch
            self.env.reset()
            self.env.reset_perturbations()
            if hasattr(self.env, "reset_disturbances"):
                self.env.reset_disturbances()

            rollout_duration = time.perf_counter() - epoch_start_time

            # Get the data from the training loop and save it
            data = buffer.get()
            if global_epoch == self.epochs:
                save_training_data(
                    self.ckpt_dir, self.training_data_file_name, data
                )

            obs = data["obs"].reshape(-1, self.env.obs_dim)
            actions = data["act"].reshape(-1, self.env.act_dim)
            rews = data["rews"].reshape(-1)
            tdres = data["tdres"].reshape(-1)
            returns = data["ret"].reshape(-1)
            logp = data["logp"].reshape(-1)
            vals = data["vals"].reshape(-1)
            if self.env.use_adaptive_approach is True:
                residuals = data["residuals"].reshape(-1, self.env.res_dim)
            else:
                residuals = jnp.empty((self.steps_per_epoch * self.env.num_envs, 0))

            obs_abs = step_outputs.prev_obs.reshape(-1, step_outputs.prev_obs.shape[-1])
            if obs_abs.size:
                tracking_error_epoch = calc_lateral_tracking_error(
                    obs_abs, self.planner
                ).mean()
                angle_error_epoch = jnp.degrees(calc_attitude_error(obs_abs)).mean()
            else:
                tracking_error_epoch = jnp.array(0.0)
                angle_error_epoch = jnp.array(0.0)

            ref_pos = jnp.atleast_2d(self.reference_point)[:, :3]
            done_events_bool = done_masks.astype(bool)
            done_count_by_env = done_events_bool.sum(axis=0)
            done_final_pos = jnp.where(
                done_events_bool[:, :, None],
                step_outputs.next_obs[:, :, :3],
                0.0,
            ).sum(axis=0)
            safe_counts = jnp.maximum(done_count_by_env, 1)[:, None]
            mean_done_final_pos = done_final_pos / safe_counts
            fallback_final_pos = step_outputs.next_obs[-1, :, :3]
            final_positions = jnp.where(
                done_count_by_env[:, None] > 0,
                mean_done_final_pos,
                fallback_final_pos,
            )
            final_pos_error_epoch = jnp.linalg.norm(
                final_positions - ref_pos, axis=1
            ).mean()

            terminals_any_epoch = jnp.any(step_outputs.success_terminals, axis=0)
            success_env_count_epoch = jnp.asarray(
                terminals_any_epoch.astype(jnp.float32).sum()
            )
            success_rate_epoch = jnp.asarray(
                terminals_any_epoch.astype(jnp.float32).mean()
            )
            if self.env.collect_reward_components:
                terminated_success = epoch_reward_components.get(
                    "terminated_success"
                )
                terminated_failure = epoch_reward_components.get(
                    "terminated_failure"
                )
                if terminated_success is not None:
                    success_termination_step_count_epoch = jnp.asarray(
                        terminated_success.astype(jnp.float32).sum()
                    )
                    success_termination_env_rate_epoch = jnp.asarray(
                        jnp.any(terminated_success > 0.0, axis=0)
                        .astype(jnp.float32)
                        .mean()
                    )
                else:
                    success_termination_step_count_epoch = jnp.array(0.0)
                    success_termination_env_rate_epoch = jnp.array(0.0)

                if terminated_failure is not None:
                    failure_termination_step_count_epoch = jnp.asarray(
                        terminated_failure.astype(jnp.float32).sum()
                    )
                    failure_termination_env_rate_epoch = jnp.asarray(
                        jnp.any(terminated_failure > 0.0, axis=0)
                        .astype(jnp.float32)
                        .mean()
                    )
                else:
                    failure_termination_step_count_epoch = jnp.array(0.0)
                    failure_termination_env_rate_epoch = jnp.array(0.0)
            else:
                success_termination_step_count_epoch = jnp.array(0.0)
                success_termination_env_rate_epoch = jnp.array(0.0)
                failure_termination_step_count_epoch = jnp.array(0.0)
                failure_termination_env_rate_epoch = jnp.array(0.0)
            terminated_step_count_epoch = jnp.asarray(
                terminated_masks.astype(jnp.float32).sum()
            )
            terminal_envs_at_end_epoch = jnp.asarray(
                step_outputs.terminals[-1].astype(jnp.float32).sum()
            )
            terminal_env_rate_at_end_epoch = jnp.asarray(
                step_outputs.terminals[-1].astype(jnp.float32).mean()
            )
            mean_ep_return_epoch = (
                jnp.asarray(episode_return_sum / episode_counter_epoch)
                if episode_counter_epoch > 0
                else jnp.array(0.0)
            )

            reward_component_means = {
                name: values.mean()
                for name, values in epoch_reward_components.items()
            }
            should_log_authority = (
                (self.env.use_wandb or self.agent.has_logger)
                and authority_logging_max_samples > 0
                and (
                    global_epoch % authority_logging_interval == 0
                    or global_epoch == self.epochs
                )
            )
            authority_payload = (
                _rollout_authority_payload(step_outputs, terminals_any_epoch)
                if should_log_authority
                else {}
            )
            reward_abs = jnp.abs(step_outputs.rewards)
            reward_scale_metrics = {
                "mean_step_reward": float(step_outputs.rewards.mean()),
                "mean_abs_step_reward": float(reward_abs.mean()),
                "max_abs_step_reward": float(reward_abs.max()),
            }
            if self.env.collect_reward_components:
                shaping_total = epoch_reward_components.get("shaping_total")
                penalty_total = epoch_reward_components.get("penalty_total")
                bonus_terminal = epoch_reward_components.get("bonus_terminal")
                if shaping_total is not None:
                    reward_scale_metrics["mean_abs_shaping_total"] = float(
                        jnp.abs(shaping_total).mean()
                    )
                if penalty_total is not None:
                    reward_scale_metrics["mean_abs_penalty_total"] = float(
                        jnp.abs(penalty_total).mean()
                    )
                if bonus_terminal is not None:
                    reward_scale_metrics["mean_terminal_bonus"] = float(
                        bonus_terminal.mean()
                    )

            if reward_scale_metrics["mean_abs_step_reward"] < 1e-4:
                print(
                    "[RewardScale] mean |step reward| < 1e-4; reward/penalty magnitudes may be too small."
                )
            # Warm up critic updates early in each failure phase
            # We scale gradients (not the optimizer) to keep the change minimal
            critic_grad_scale = 1.0
            if (
                self.env.train_with_failures
                and active_failures
                and phase_epoch < critic_warmup_epochs
            ):
                critic_grad_scale = critic_warmup_scale

            update_start_time = time.perf_counter()
            (
                last_actor_loss,
                last_critic_loss,
                mean_actor_loss,
                mean_critic_loss,
                last_kl,
                mean_kl,
                mean_clip_frac,
            ) = self.agent.update_actor_critic_minibatch(
                actor_key,
                jnp.concatenate([obs, residuals], axis=1),
                actions,
                tdres,
                logp,
                returns,
                vals,
                critic_grad_scale=critic_grad_scale,
            )
            update_duration = time.perf_counter() - update_start_time

            actor_loss_last_f = float(last_actor_loss)
            critic_loss_last_f = float(last_critic_loss)
            actor_loss_mean_f = float(mean_actor_loss)
            critic_loss_mean_f = float(mean_critic_loss)
            kl_last_f = float(last_kl)
            kl_mean_f = float(mean_kl)
            clip_frac_f = float(mean_clip_frac)
            mean_std_f = float(jnp.exp(self.agent.actor.log_std.value).mean())
            mean_log_std_f = float(self.agent.actor.log_std.value.mean())

            adv_mean_f = float(tdres.mean())
            adv_std_f = float(tdres.std())
            value_mean_f = float(vals.mean())
            value_std_f = float(vals.std())
            return_mean_f = float(returns.mean())
            return_std_f = float(returns.std())
            critic_loss_normalized_f = float(
                critic_loss_mean_f / (return_std_f * return_std_f + 1e-8)
            )
            var_returns = jnp.var(returns)
            explained_var = jnp.where(
                var_returns > 1e-8,
                1.0 - (jnp.var(returns - vals) / var_returns),
                0.0,
            )
            explained_var_f = float(explained_var)

            logging_start_time = time.perf_counter()
            # Monitor key RL metrics during training using Weights & Biases
            if self.env.use_wandb:
                wandb_payload = build_wandb_policy_training_payload(
                    phase_name=phase_name,
                    phase_epoch=phase_epoch + 1,
                    global_epoch=global_epoch,
                    critic_grad_scale=float(critic_grad_scale),
                    actor_loss_mean=actor_loss_mean_f,
                    critic_loss_mean=critic_loss_mean_f,
                    true_kl_mean=kl_mean_f,
                    clip_fraction=clip_frac_f,
                    explained_variance=explained_var_f,
                    return_std=return_std_f,
                    value_std=value_std_f,
                    critic_loss_normalized=critic_loss_normalized_f,
                    mean_std=mean_std_f,
                    mean_log_std=mean_log_std_f,
                    mean_episodic_returns=float(mean_ep_return_epoch),
                    success_env_count=float(success_env_count_epoch),
                    success_rate=float(success_rate_epoch),
                    terminated_step_count=float(terminated_step_count_epoch),
                    success_termination_step_count=float(
                        success_termination_step_count_epoch
                    ),
                    failure_termination_step_count=float(
                        failure_termination_step_count_epoch
                    ),
                    success_termination_env_rate=float(
                        success_termination_env_rate_epoch
                    ),
                    failure_termination_env_rate=float(
                        failure_termination_env_rate_epoch
                    ),
                    terminal_envs_at_end=float(terminal_envs_at_end_epoch),
                    terminal_env_rate_at_end=float(
                        terminal_env_rate_at_end_epoch
                    ),
                    mean_lateral_error=float(tracking_error_epoch),
                    mean_angle_error=float(angle_error_epoch),
                    mean_final_position_error=float(final_pos_error_epoch),
                    reward_component_means={
                        metric_name: float(metric_value)
                        for metric_name, metric_value in reward_component_means.items()
                    },
                    reward_scale_metrics=reward_scale_metrics,
                )
                wandb_payload.update(authority_payload)
                wandb.log(
                    wandb_payload,
                    step=global_epoch,
                )

            # Log key RL metrics
            if self.agent.has_logger:
                logger_payload = build_logger_policy_training_payload(
                    phase_name=phase_name,
                    phase_epoch=int(phase_epoch + 1),
                    critic_grad_scale=float(critic_grad_scale),
                    actor_loss_mean=actor_loss_mean_f,
                    critic_loss_mean=critic_loss_mean_f,
                    true_kl_mean=kl_mean_f,
                    clip_fraction=clip_frac_f,
                    explained_variance=explained_var_f,
                    return_std=return_std_f,
                    value_std=value_std_f,
                    critic_loss_normalized=critic_loss_normalized_f,
                    success_env_count=float(success_env_count_epoch),
                    success_rate=float(success_rate_epoch),
                    terminated_step_count=float(terminated_step_count_epoch),
                    success_termination_step_count=float(
                        success_termination_step_count_epoch
                    ),
                    failure_termination_step_count=float(
                        failure_termination_step_count_epoch
                    ),
                    success_termination_env_rate=float(
                        success_termination_env_rate_epoch
                    ),
                    failure_termination_env_rate=float(
                        failure_termination_env_rate_epoch
                    ),
                    terminal_envs_at_end=float(terminal_envs_at_end_epoch),
                    terminal_env_rate_at_end=float(
                        terminal_env_rate_at_end_epoch
                    ),
                    mean_std=mean_std_f,
                    mean_lateral_error=float(tracking_error_epoch),
                    mean_angle_error=float(angle_error_epoch),
                    mean_final_position_error=float(final_pos_error_epoch),
                    train_mean_episodic_returns=float(mean_ep_return_epoch),
                    reward_scale_metrics=reward_scale_metrics,
                )
                logger_payload.update(authority_payload)
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    step=int(global_epoch),
                    run_name=self.env.run_name,
                    stage="policy_training",
                    **logger_payload,
                )
            logging_duration = time.perf_counter() - logging_start_time

            # Safeguard: periodically evaluate and keep the best nominal checkpoint
            eval_duration = 0.0
            eval_due = bool(active_failures) and (
                (phase_epoch + 1) % eval_interval == 0
                or phase_epoch == phase_epochs - 1
            )
            if eval_due:
                eval_start_time = time.perf_counter()
                eval_keys = jax.random.split(eval_key, 3)
                nominal_score = _eval_policy(
                    key=eval_keys[0],
                    fraction_perturbed_envs=0.0,
                    perturbation_distribution=zeros_dist,
                    disturbance_fraction=0.0,
                )

                # Evaluate the newest failure in isolation
                failure_dist = zeros_dist
                if new_failure_idx is not None:
                    failure_dist = failure_dist.at[int(new_failure_idx)].set(1.0)
                failure_score = _eval_policy(
                    key=eval_keys[1],
                    fraction_perturbed_envs=1.0,
                    perturbation_distribution=failure_dist,
                    disturbance_fraction=0.0,
                )

                mixture_score = _eval_policy(
                    key=eval_keys[2],
                    fraction_perturbed_envs=phase_failure_fraction,
                    perturbation_distribution=phase_distribution,
                    disturbance_fraction=phase_disturbance_fraction,
                )

                if nominal_score > best_nominal_score:
                    best_nominal_score = nominal_score
                    best_actor_state = nnx.state(self.agent.actor)
                    best_critic_state = nnx.state(self.agent.critic)

                print(
                    f"[Curriculum Eval] phase={phase_name} epoch={phase_epoch + 1}/{phase_epochs} "
                    f"nominal={nominal_score:.4f} failure_k={failure_score:.4f} mixture={mixture_score:.4f} "
                    f"best_nominal={best_nominal_score:.4f}"
                )
                eval_duration = time.perf_counter() - eval_start_time

            # Save the trained actor and critic network weights
            save_start_time = time.perf_counter()
            should_save_checkpoint = (
                global_epoch % checkpoint_interval == 0
                or global_epoch == self.epochs
            )
            if should_save_checkpoint:
                save_trained_modules(
                    self.agent, self.ckpt_dir, self.training_state_file_name
                )
            save_duration = time.perf_counter() - save_start_time
            epoch_total_duration = time.perf_counter() - epoch_start_time
            epoch_timing = EpochTiming(
                rollout=rollout_duration,
                scan=scan_time,
                buffer=buffer_time,
                update=update_duration,
                logging=logging_duration,
                eval=eval_duration,
                save_ckpt=save_duration,
                total=epoch_total_duration,
            )
            print(
                format_epoch_timing_line(
                    global_epoch=global_epoch,
                    total_epochs=self.epochs,
                    phase_name=phase_name,
                    phase_epoch=phase_epoch + 1,
                    phase_epochs=phase_epochs,
                    timing=epoch_timing,
                )
            )

    # Restore the best nominal checkpoint at the end of the curriculum
    if best_actor_state is not None and best_critic_state is not None:
        nnx.update(self.agent.actor, best_actor_state)
        nnx.update(self.agent.critic, best_critic_state)
        save_trained_modules(
            self.agent, self.ckpt_dir, self.training_state_file_name
        )
