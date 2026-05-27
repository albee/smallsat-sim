import os
import time
from collections.abc import Mapping

import jax
import jax.numpy as jnp
from flax import nnx
import wandb

from smallsat_sim.controllers.rl.runners.rollout import (
    FunctionalRolloutCallbacks,
    prepare_policy_input_with_residuals,
    residuals_from_wrench_delta,
    run_functional_rollout,
)
from smallsat_sim.controllers.rl.runners.curriculum import (
    build_authority_regime_curriculum,
    build_authority_regime_curriculum_v2,
    uniform_failure_distribution,
)
from smallsat_sim.controllers.rl.runners.failure_scenarios import (
    SPLIT_SEMANTIC_EVAL,
    SPLIT_STRESS_TEST,
    SPLIT_TRAIN,
    apply_sampled_failure_scenario_split,
    build_failure_scenario_table,
    precompute_scenario_gp_samples,
    scenario_authority_regime_counts,
    scenario_bin_counts,
    scenario_split_counts,
    scenario_task_regime_counts,
    save_scenario_table_csv,
    task_wrench_from_state_features,
)
from smallsat_sim.controllers.rl.runners.runner_timing import (
    EpochTiming,
    format_epoch_timing_line,
)
from smallsat_sim.controllers.rl.runners.training_helpers import (
    evaluate_policy_checkpoint,
    rollout_authority_payload,
    sample_start_time,
)
from smallsat_sim.controllers.rl.runners.runner_metrics import (
    build_logger_policy_training_payload,
    build_wandb_policy_training_payload,
)
from smallsat_sim.controllers.rl.runners.runner_utils import (
    load_trained_modules,
    save_training_data,
    save_trained_modules,
    update_module_from_checkpoint_state,
)
from smallsat_sim.controllers.rl.storage.replay_buffer import ReplayBuffer
from smallsat_sim.envs.perturbations_rl import Perturbation
from smallsat_sim.envs.vec_env import (
    _compute_state_features,
    _compute_freeflyer_state_features,
    freeflyer_reset,
    freeflyer_reset_masked,
    vecenv_step_training,
    vecenv_step_training_freeflyer,
)


def _context_input_weight_norm(module, obs_dim: int, res_dim: int) -> float:
    """
    Return the largest first-layer weight norm attached to adaptive context inputs.
    """
    if res_dim <= 0:
        return 0.0

    norms = []

    def visit(node) -> None:
        if hasattr(node, "value"):
            value = node.value
            if (
                hasattr(value, "shape")
                and len(value.shape) == 2
                and int(value.shape[0]) == int(obs_dim + res_dim)
                and int(value.shape[0]) > int(obs_dim)
            ):
                norms.append(jnp.linalg.norm(value[obs_dim:, :]))
            return
        if isinstance(node, Mapping):
            for child in node.values():
                visit(child)

    visit(nnx.state(module))
    if not norms:
        return 0.0
    return float(jnp.max(jnp.stack(norms)))


def learn_runner(self) -> None:
    """
    Main training loop.
    """
    # Check if training has already been done
    file_path = os.path.join(self.ckpt_dir, self.training_state_file_name)
    if os.path.isfile(file_path):
        return

    def _load_actor_critic_checkpoint(ckpt_filename: str) -> bool:
        file_path = os.path.join(self.ckpt_dir, ckpt_filename)
        if not os.path.isfile(file_path):
            return False
        restored_state = load_trained_modules(self.ckpt_dir, ckpt_filename)
        actor_state = restored_state["actor_model"]
        critic_state = restored_state["critic_model"]
        update_module_from_checkpoint_state(self.agent.actor, actor_state)
        update_module_from_checkpoint_state(self.agent.critic, critic_state)
        return True

    if self.env.use_pretrained:
        # Check if pretrained actor and critic modules are available and load them
        if _load_actor_critic_checkpoint(self.pretraining_state_file_name):
            print(f"Loaded pretrained checkpoint {self.pretraining_state_file_name}.\n")
        else:
            print("No pretrained modules available.\n")
    elif self.env.train_with_failures:
        print("Training failure policy from scratch with native input dimensions.\n")

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
    failure_ramp_epochs = max(1, int(getattr(cfg, "curriculum_failure_ramp_epochs", 1)))
    disturbance_ramp_epochs = max(
        1, int(getattr(cfg, "curriculum_disturbance_ramp_epochs", 1))
    )
    resample_effects_interval = max(
        1, int(getattr(cfg, "curriculum_resample_effects_interval", 1))
    )
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
    curriculum_mode = str(getattr(cfg, "failure_curriculum_mode", "authority"))
    use_controllable_failure_scenarios = bool(
        getattr(cfg, "use_controllable_failure_scenarios", False)
    ) and bool(self.env.train_with_failures)
    split_name_to_id = {
        "train": SPLIT_TRAIN,
        "semantic_eval": SPLIT_SEMANTIC_EVAL,
        "stress_test": SPLIT_STRESS_TEST,
    }
    failure_scenario_train_split = str(
        getattr(cfg, "failure_scenario_train_split", "train")
    )
    if failure_scenario_train_split == "stress":
        print(
            "[Failure Scenarios] split='stress' is deprecated (stress is now a label). "
            "Using split='train'.",
            flush=True,
        )
        failure_scenario_train_split = "train"
    failure_scenario_split_id = split_name_to_id.get(
        failure_scenario_train_split, SPLIT_TRAIN
    )
    failure_scenario_table = None
    use_task_conditioned_failure_sampling = bool(
        getattr(cfg, "use_task_conditioned_failure_sampling", True)
    )
    if use_controllable_failure_scenarios:
        failure_scenario_table = build_failure_scenario_table(
            self.env._thruster_mixer_T,
            self.agent.actor.act_low,
            self.agent.actor.act_high,
            max_faults=int(getattr(cfg, "failure_scenario_max_faults", 12)),
            exhaustive_faults=int(
                getattr(cfg, "failure_scenario_exhaustive_faults", 2)
            ),
            sampled_per_fault_count=int(
                getattr(cfg, "failure_scenario_sampled_per_fault_count", 512)
            ),
            min_rank=int(getattr(cfg, "failure_scenario_min_rank", 6)),
            stress_quantile=float(
                getattr(cfg, "failure_scenario_stress_quantile", 0.9)
            ),
            mild_effectiveness=float(
                getattr(cfg, "failure_scenario_mild_effectiveness", 0.5)
            ),
        )
        print(
            "[Failure Scenarios] Using controllability-filtered scenario table "
            f"{scenario_split_counts(failure_scenario_table)}; "
            f"training split={failure_scenario_train_split}; "
            f"train bins={scenario_bin_counts(failure_scenario_table)}; "
            f"train authority regimes="
            f"{scenario_authority_regime_counts(failure_scenario_table, SPLIT_TRAIN)}; "
            f"train task regimes="
            f"{scenario_task_regime_counts(failure_scenario_table, SPLIT_TRAIN)}"
        )
        scenario_table_path = os.path.join(
            self.ckpt_dir,
            f"failure_scenarios_{self.training_state_file_name.replace('.pkl', '.csv')}",
        )
        save_scenario_table_csv(failure_scenario_table, scenario_table_path)
        print(f"[Failure Scenarios] Saved scenario table to {scenario_table_path}")
        precompute_start = time.perf_counter()
        precompute_scenario_gp_samples(self.env, self._take_keys())
        print(
            "[Failure Scenarios] Precomputed GP failure samples in "
            f"{time.perf_counter() - precompute_start:.2f}s"
        )
    # Keep evaluation lightweight relative to training rollouts.
    eval_episodes = max(1, min(self.n_evals, 3))

    if curriculum_mode not in ("authority", "semantic"):
        print(
            f"[Curriculum] mode='{curriculum_mode}' is deprecated; using semantic authority curriculum.",
            flush=True,
        )
    phases, total_epochs = build_authority_regime_curriculum_v2(
        train_with_failures=bool(self.env.train_with_failures),
        fallback_epochs=int(cfg.PPO.epochs),
        nominal_epochs=nominal_epochs,
        phase_epochs=phase_epochs,
        failure_fraction=failure_fraction,
        disturbance_fraction=disturbance_fraction,
    )
    # Align runner/agent epoch counts with the curriculum length
    self.epochs = total_epochs
    if hasattr(self.agent, "epochs"):
        self.agent.epochs = total_epochs

    global_epoch = 0
    # Convenience distribution for nominal-only evaluation.
    zeros_dist = jnp.zeros((5,), dtype=jnp.float32)

    # Reuse step-config objects across epochs. Rebuilding them every epoch can
    # force extra trace/compile work in the scan path because the config carries
    # large pytrees (MJX templates and effect snapshots).
    step_config_cache: dict[bool, object] = {}

    for phase_idx, phase in enumerate(phases):
        phase_name = phase["name"]
        phase_epochs = int(phase["epochs"])
        active_failures = list(phase["active_failures"])
        phase_failure_fraction = float(phase["failure_fraction"])
        phase_disturbance_fraction = float(phase["disturbance_fraction"])
        phase_distribution = uniform_failure_distribution(active_failures)
        new_failure_idx = phase["new_failure"]
        phase_difficulty_bin = phase.get("difficulty_bin")
        current_difficulty_bin = -1 if phase_difficulty_bin is None else int(
            phase_difficulty_bin
        )
        phase_authority_regime = phase.get("authority_regime")
        current_authority_regime = (
            -1 if phase_authority_regime is None else int(phase_authority_regime)
        )
        phase_task_feasibility_regime = phase.get("task_feasibility_regime")
        current_task_feasibility_regime = (
            -1
            if phase_task_feasibility_regime is None
            else int(phase_task_feasibility_regime)
        )
        phase_authority_label_any_mask = phase.get("authority_label_any_mask")
        phase_failure_sampling_mix = phase.get("failure_sampling_mix")
        phase_uses_perturbations = bool(active_failures) and phase_failure_fraction > 0.0
        phase_uses_disturbances = phase_disturbance_fraction > 0.0
        phase_uses_effects = phase_uses_perturbations or phase_uses_disturbances
        applied_failure_fraction = 0.0
        applied_disturbance_fraction = 0.0
        active_scenario_payload: dict[str, float] = {}

        print(
            f"[Curriculum] Phase {phase_idx + 1}/{len(phases)}: {phase_name} "
            f"for {phase_epochs} epochs (failure_fraction={phase_failure_fraction:.2f}, "
            f"disturbance_fraction={phase_disturbance_fraction:.2f})"
        )

        for phase_epoch in range(phase_epochs):
            global_epoch += 1
            epoch_start_time = time.perf_counter()
            setup_start_time = time.perf_counter()
            failure_ramp = min(1.0, float(phase_epoch + 1) / float(failure_ramp_epochs))
            disturbance_ramp = min(
                1.0, float(phase_epoch + 1) / float(disturbance_ramp_epochs)
            )
            current_failure_fraction = phase_failure_fraction * failure_ramp
            current_disturbance_fraction = phase_disturbance_fraction * disturbance_ramp
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
            failure_start_time = sample_start_time(
                failure_onset_key,
                failure_start_time_min,
                failure_start_time_max,
            )
            disturbance_start_time = sample_start_time(
                disturbance_onset_key,
                disturbance_start_time_min,
                disturbance_start_time_max,
            )

            # Apply failures/disturbances for this phase with fixed proportions
            should_resample_effects = (
                phase_epoch == 0
                or phase_epoch % resample_effects_interval == 0
            )
            if self.env.train_with_failures and phase_uses_effects:
                if phase_uses_perturbations:
                    if should_resample_effects:
                        applied_failure_fraction = current_failure_fraction
                        self.env.reset_perturbations()
                        if (
                            use_controllable_failure_scenarios
                            and failure_scenario_table is not None
                        ):
                            current_task_wrenches = None
                            if use_task_conditioned_failure_sampling:
                                current_states = _compute_state_features(
                                    self.env.state_struct.mjx_batch,
                                    self.reference_point,
                                )
                                pd_gains = self.env.env_cfg.control.PD.gains
                                current_task_wrenches = task_wrench_from_state_features(
                                    current_states,
                                    kp_pos=float(getattr(pd_gains, "Kp_x", 0.2)),
                                    kd_pos=float(getattr(pd_gains, "Kd_x", 1.0)),
                                    kp_att=float(getattr(pd_gains, "Kp_q", 3.0)),
                                    kd_att=float(getattr(pd_gains, "Kd_q", 5.0)),
                                )
                            (
                                active_scenario_payload,
                                _selected_envs,
                                _scenario_indices,
                            ) = apply_sampled_failure_scenario_split(
                                self.env,
                                key=perturb_key,
                                table=failure_scenario_table,
                                split_id=failure_scenario_split_id,
                                fraction_perturbed_envs=applied_failure_fraction,
                                start_time=failure_start_time,
                                difficulty_bin=phase_difficulty_bin,
                                authority_regime=phase_authority_regime,
                                task_feasibility_regime=phase_task_feasibility_regime,
                                authority_label_any_mask=phase_authority_label_any_mask,
                                failure_sampling_mix=phase_failure_sampling_mix,
                                task_wrenches=current_task_wrenches,
                                return_selection=True,
                            )
                        else:
                            self.env.apply_random_perturbations(
                                key=perturb_key,
                                fraction_perturbed_envs=applied_failure_fraction,
                                perturbation_distribution=phase_distribution,
                                start_time=failure_start_time,
                            )
                if phase_uses_disturbances:
                    if should_resample_effects:
                        applied_disturbance_fraction = current_disturbance_fraction
                        if hasattr(self.env, "reset_disturbances"):
                            self.env.reset_disturbances()
                        self.env.apply_random_disturbance(
                            key=disturb_key,
                            fraction_disturbed_envs=applied_disturbance_fraction,
                            start_time=disturbance_start_time,
                        )
            has_perturbation_epoch, has_disturbance_epoch = self.env._get_active_failure_masks()
            failed_env_mask_epoch = jnp.logical_or(
                has_perturbation_epoch, has_disturbance_epoch
            )
            nominal_env_mask_epoch = jnp.logical_not(failed_env_mask_epoch)

            # Accumulate rollout stats to emit once per epoch
            setup_duration = time.perf_counter() - setup_start_time
            scan_start = time.perf_counter()
            if phase_uses_effects not in step_config_cache:
                step_config_cache[phase_uses_effects] = self.env.build_step_config(
                    max_episode_len=self.max_ep_len,
                    effects_enabled=phase_uses_effects,
                )
            step_config = step_config_cache[phase_uses_effects]
            rollout_backend = os.environ.get(
                "SMALLSAT_ROLLOUT_BACKEND",
                getattr(self.env.env_cfg.control.RL, "rollout_backend", "mjx"),
            )
            if rollout_backend == "freeflyer":
                initial_state = self.env.freeflyer_state_struct()
                rollout_step_fn = vecenv_step_training_freeflyer
                rollout_reset_fn = freeflyer_reset_masked
                rollout_state_features_fn = _compute_freeflyer_state_features
            else:
                initial_state = self.env.state_struct
                rollout_step_fn = vecenv_step_training
                rollout_reset_fn = None
                rollout_state_features_fn = _compute_state_features
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
                del _step, reset_flag  # unused
                desired_wrench = (
                    actions @ self.env._thruster_mixer_T
                    if self.env.use_adaptive_approach
                    else None
                )
                residuals_next = residuals_from_wrench_delta(
                    step_output=step_output,
                    residuals=residuals,
                    use_adaptive_approach=self.env.use_adaptive_approach,
                    adaptive_context_mode=self.env.adaptive_context_mode,
                    thruster_mixer_T=self.env._thruster_mixer_T,
                    commanded_ctrl=actions,
                    desired_wrench=desired_wrench,
                )
                return residuals_next, None, carry_extra

            def _bootstrap_value(
                step_idx, env_state, residuals, rng_key, carry_extra
            ):
                rng_key, value_key = jax.random.split(rng_key)
                # Bootstrap with the critic on the next observation
                if rollout_backend == "freeflyer":
                    next_states = _compute_freeflyer_state_features(
                        env_state, self.reference_point
                    )
                else:
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
                step_fn=rollout_step_fn,
                reset_fn=rollout_reset_fn,
                state_features_fn=rollout_state_features_fn,
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
                and rollout_backend == "mjx"
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

            sync_start_time = time.perf_counter()
            if self.agent.has_logger and rollout_backend == "mjx":
                # Local per-episode logging reads the imperative env timestamp.
                self.env.apply_state_struct(rollout_result.final_state)
            else:
                # The next operation is a full env reset, so avoid copying the
                # full MJX batch back to the imperative env. Preserve only RNG
                # progression so epoch-to-epoch randomization stays identical.
                self.env._rng = rollout_result.final_state.rng
                if hasattr(rollout_result.final_state, "time"):
                    self.env.mjx_batch = self.env.mjx_batch.replace(
                        time=rollout_result.final_state.time
                    )
                self.env._state = self.env._state.replace(
                    rng=rollout_result.final_state.rng
                )
            sync_duration = time.perf_counter() - sync_start_time

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
            done_returns = episode_returns_traj * done_events
            episode_return_sum_epoch = done_returns.sum()
            episode_counter_epoch = done_events.sum()

            if self.agent.has_logger:
                if float(episode_counter_epoch) > 0.0:
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

            epoch_reward_components = step_outputs.reward_components

            # Reset the imperative environment for the next epoch
            reset_start_time = time.perf_counter()
            if rollout_backend == "freeflyer":
                reset_freeflyer_state = freeflyer_reset(
                    self.env._rng,
                    config=step_config,
                )
                jax.block_until_ready(reset_freeflyer_state.qpos)
                self.env._rng = reset_freeflyer_state.rng
                self.env._state = self.env._state.replace(rng=reset_freeflyer_state.rng)
            else:
                self.env.reset()
                jax.block_until_ready(self.env.mjx_batch.qpos)
            reset_duration = time.perf_counter() - reset_start_time

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

            tracking_error_epoch = jnp.linalg.norm(
                step_outputs.prev_states[:, :, :3], axis=2
            ).mean()
            angle_error_epoch = jnp.degrees(
                jnp.linalg.norm(step_outputs.prev_states[:, :, 3:6], axis=2)
            ).mean()

            done_events_bool = done_masks.astype(bool)
            done_count_by_env = done_events_bool.sum(axis=0)
            done_final_pos_error = jnp.where(
                done_events_bool[:, :, None],
                step_outputs.next_position_error,
                0.0,
            ).sum(axis=0)
            safe_counts = jnp.maximum(done_count_by_env, 1)[:, None]
            mean_done_final_error = done_final_pos_error / safe_counts
            fallback_final_error = step_outputs.next_position_error[-1]
            final_position_errors = jnp.where(
                done_count_by_env[:, None] > 0,
                mean_done_final_error,
                fallback_final_error,
            )
            final_pos_error_by_env = jnp.linalg.norm(final_position_errors, axis=1)
            final_pos_error_epoch = final_pos_error_by_env.mean()
            median_final_pos_error_epoch = jnp.median(final_pos_error_by_env)
            p75_final_pos_error_epoch = jnp.quantile(final_pos_error_by_env, 0.75)
            p90_final_pos_error_epoch = jnp.quantile(final_pos_error_by_env, 0.90)

            def _final_scalar_from_done(metric_seq):
                done_metric_sum = jnp.where(done_events_bool, metric_seq, 0.0).sum(
                    axis=0
                )
                mean_done_metric = done_metric_sum / jnp.maximum(
                    done_count_by_env, 1
                )
                return jnp.where(done_count_by_env > 0, mean_done_metric, metric_seq[-1])

            final_speed_by_env = _final_scalar_from_done(step_outputs.next_speed)
            final_att_error_by_env = _final_scalar_from_done(
                step_outputs.next_attitude_error
            )
            final_ang_speed_by_env = _final_scalar_from_done(
                step_outputs.next_angular_speed
            )
            final_pos_ok_by_env = final_pos_error_by_env <= self.env.terminal_radius
            final_speed_ok_by_env = final_speed_by_env <= self.env.terminal_max_speed
            final_att_ok_by_env = (
                final_att_error_by_env <= self.env.terminal_max_att_error
            )
            final_ang_speed_ok_by_env = (
                final_ang_speed_by_env <= self.env.terminal_max_ang_speed
            )
            fraction_pos_within_radius_epoch = final_pos_ok_by_env.astype(
                jnp.float32
            ).mean()
            fraction_speed_within_limit_epoch = final_speed_ok_by_env.astype(
                jnp.float32
            ).mean()
            fraction_att_within_limit_epoch = final_att_ok_by_env.astype(
                jnp.float32
            ).mean()
            fraction_ang_speed_within_limit_epoch = final_ang_speed_ok_by_env.astype(
                jnp.float32
            ).mean()
            fraction_all_conditions_except_hold_epoch = (
                final_pos_ok_by_env
                & final_speed_ok_by_env
                & final_att_ok_by_env
                & final_ang_speed_ok_by_env
            ).astype(jnp.float32).mean()

            pos_ok_seq = (
                jnp.linalg.norm(step_outputs.next_position_error, axis=-1)
                <= self.env.terminal_radius
            )

            def _hold_count_scan(carry, inputs):
                counts, max_counts = carry
                pos_ok, done = inputs
                next_counts = jnp.where(pos_ok, counts + 1, 0)
                next_max_counts = jnp.maximum(max_counts, next_counts)
                carry_counts = jnp.where(done, 0, next_counts)
                return (carry_counts, next_max_counts), None

            (_, max_success_hold_steps_by_env), _ = jax.lax.scan(
                _hold_count_scan,
                (
                    jnp.zeros((self.env.num_envs,), dtype=jnp.int32),
                    jnp.zeros((self.env.num_envs,), dtype=jnp.int32),
                ),
                (pos_ok_seq, done_events_bool),
            )
            mean_consecutive_success_hold_steps_epoch = (
                max_success_hold_steps_by_env.astype(jnp.float32).mean()
            )

            terminals_any_epoch = jnp.any(step_outputs.success_terminals, axis=0)
            success_env_count_epoch = jnp.asarray(
                terminals_any_epoch.astype(jnp.float32).sum()
            )
            success_rate_epoch = jnp.asarray(
                terminals_any_epoch.astype(jnp.float32).mean()
            )
            nominal_env_count_epoch = jnp.asarray(
                nominal_env_mask_epoch.astype(jnp.float32).sum()
            )
            failed_env_count_epoch = jnp.asarray(
                failed_env_mask_epoch.astype(jnp.float32).sum()
            )
            nominal_success_rate_epoch = jnp.where(
                nominal_env_count_epoch > 0.0,
                (
                    terminals_any_epoch.astype(jnp.float32)
                    * nominal_env_mask_epoch.astype(jnp.float32)
                ).sum()
                / nominal_env_count_epoch,
                0.0,
            )
            failed_success_rate_epoch = jnp.where(
                failed_env_count_epoch > 0.0,
                (
                    terminals_any_epoch.astype(jnp.float32)
                    * failed_env_mask_epoch.astype(jnp.float32)
                ).sum()
                / failed_env_count_epoch,
                0.0,
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
            mean_ep_return_epoch = jnp.where(
                episode_counter_epoch > 0.0,
                episode_return_sum_epoch / jnp.maximum(episode_counter_epoch, 1.0),
                0.0,
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
                rollout_authority_payload(
                    self,
                    step_outputs=step_outputs,
                    actions_traj=actions_traj,
                    success_by_env=terminals_any_epoch,
                    max_samples=authority_logging_max_samples,
                )
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
            adaptive_context_payload = {}
            if self.env.use_adaptive_approach and self.env.res_dim > 0:
                adaptive_context_payload = {
                    "adaptive_context/mean_abs": float(jnp.abs(residuals).mean()),
                    "adaptive_context/std": float(residuals.std()),
                    "adaptive_context/residual_wrench_norm_mean": float(
                        jnp.linalg.norm(residuals[:, : min(6, self.env.res_dim)], axis=1).mean()
                    ),
                    "adaptive_context/actor_context_input_weight_norm": _context_input_weight_norm(
                        self.agent.actor,
                        self.env.obs_dim,
                        self.env.res_dim,
                    ),
                    "adaptive_context/critic_context_input_weight_norm": _context_input_weight_norm(
                        self.agent.critic,
                        self.env.obs_dim,
                        self.env.res_dim,
                    ),
                }
                if self.env.res_dim >= 12:
                    adaptive_context_payload[
                        "adaptive_context/bias_wrench_norm_mean"
                    ] = float(jnp.linalg.norm(residuals[:, 6:12], axis=1).mean())

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
                    nominal_env_count=float(nominal_env_count_epoch),
                    failed_env_count=float(failed_env_count_epoch),
                    nominal_success_rate=float(nominal_success_rate_epoch),
                    failed_success_rate=float(failed_success_rate_epoch),
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
                    current_failure_fraction=float(applied_failure_fraction),
                    current_disturbance_fraction=float(applied_disturbance_fraction),
                    current_difficulty_bin=current_difficulty_bin,
                    current_authority_regime=current_authority_regime,
                    current_task_feasibility_regime=current_task_feasibility_regime,
                    median_final_position_error=float(
                        median_final_pos_error_epoch
                    ),
                    p75_final_position_error=float(p75_final_pos_error_epoch),
                    p90_final_position_error=float(p90_final_pos_error_epoch),
                    fraction_pos_within_radius=float(
                        fraction_pos_within_radius_epoch
                    ),
                    fraction_speed_within_limit=float(
                        fraction_speed_within_limit_epoch
                    ),
                    fraction_att_within_limit=float(
                        fraction_att_within_limit_epoch
                    ),
                    fraction_ang_speed_within_limit=float(
                        fraction_ang_speed_within_limit_epoch
                    ),
                    fraction_all_conditions_except_hold=float(
                        fraction_all_conditions_except_hold_epoch
                    ),
                    mean_consecutive_success_hold_steps=float(
                        mean_consecutive_success_hold_steps_epoch
                    ),
                    reward_component_means={
                        metric_name: float(metric_value)
                        for metric_name, metric_value in reward_component_means.items()
                    },
                    reward_scale_metrics=reward_scale_metrics,
                )
                wandb_payload.update(authority_payload)
                wandb_payload.update(active_scenario_payload)
                wandb_payload.update(adaptive_context_payload)
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
                    nominal_env_count=float(nominal_env_count_epoch),
                    failed_env_count=float(failed_env_count_epoch),
                    nominal_success_rate=float(nominal_success_rate_epoch),
                    failed_success_rate=float(failed_success_rate_epoch),
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
                    current_failure_fraction=float(applied_failure_fraction),
                    current_disturbance_fraction=float(applied_disturbance_fraction),
                    current_difficulty_bin=current_difficulty_bin,
                    current_authority_regime=current_authority_regime,
                    current_task_feasibility_regime=current_task_feasibility_regime,
                    median_final_position_error=float(
                        median_final_pos_error_epoch
                    ),
                    p75_final_position_error=float(p75_final_pos_error_epoch),
                    p90_final_position_error=float(p90_final_pos_error_epoch),
                    fraction_pos_within_radius=float(
                        fraction_pos_within_radius_epoch
                    ),
                    fraction_speed_within_limit=float(
                        fraction_speed_within_limit_epoch
                    ),
                    fraction_att_within_limit=float(
                        fraction_att_within_limit_epoch
                    ),
                    fraction_ang_speed_within_limit=float(
                        fraction_ang_speed_within_limit_epoch
                    ),
                    fraction_all_conditions_except_hold=float(
                        fraction_all_conditions_except_hold_epoch
                    ),
                    mean_consecutive_success_hold_steps=float(
                        mean_consecutive_success_hold_steps_epoch
                    ),
                    train_mean_episodic_returns=float(mean_ep_return_epoch),
                    reward_scale_metrics=reward_scale_metrics,
                )
                logger_payload.update(authority_payload)
                logger_payload.update(active_scenario_payload)
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    step=int(global_epoch),
                    run_name=self.env.run_name,
                    stage="policy_training",
                    **logger_payload,
                )
            logging_duration = time.perf_counter() - logging_start_time

            eval_duration = 0.0
            eval_due = bool(active_failures) and (
                (phase_epoch + 1) % eval_interval == 0
                or phase_epoch == phase_epochs - 1
            )
            if eval_due:
                eval_start_time = time.perf_counter()
                eval_keys = jax.random.split(eval_key, 3)
                # In-loop eval mutates the shared env object (resets + applies effects).
                # Snapshot and restore training state so eval cannot leak into the next epoch.
                train_state_snapshot = self.env.state_struct
                train_perturbation_states_snapshot = getattr(
                    self.env, "perturbation_states", None
                )
                train_disturbance_states_snapshot = getattr(
                    self.env, "disturbance_states", None
                )
                train_thruster_mask_snapshot = getattr(Perturbation, "thruster_mask", None)
                try:
                    nominal_score = evaluate_policy_checkpoint(
                        self,
                        key=eval_keys[0],
                        fraction_perturbed_envs=0.0,
                        perturbation_distribution=zeros_dist,
                        disturbance_fraction=0.0,
                        eval_episodes=eval_episodes,
                    )

                    # Evaluate the newest failure in isolation
                    failure_dist = zeros_dist
                    if new_failure_idx is not None:
                        failure_dist = failure_dist.at[int(new_failure_idx)].set(1.0)
                    failure_score = evaluate_policy_checkpoint(
                        self,
                        key=eval_keys[1],
                        fraction_perturbed_envs=1.0,
                        perturbation_distribution=failure_dist,
                        disturbance_fraction=0.0,
                        eval_episodes=eval_episodes,
                    )

                    mixture_score = evaluate_policy_checkpoint(
                        self,
                        key=eval_keys[2],
                        fraction_perturbed_envs=applied_failure_fraction,
                        perturbation_distribution=phase_distribution,
                        disturbance_fraction=applied_disturbance_fraction,
                        eval_episodes=eval_episodes,
                    )
                finally:
                    self.env._state = train_state_snapshot
                    if train_perturbation_states_snapshot is not None:
                        self.env.perturbation_states = train_perturbation_states_snapshot
                    if train_disturbance_states_snapshot is not None:
                        self.env.disturbance_states = train_disturbance_states_snapshot
                    if train_thruster_mask_snapshot is not None:
                        Perturbation.thruster_mask = train_thruster_mask_snapshot

                print(
                    f"[Curriculum Eval] phase={phase_name} epoch={phase_epoch + 1}/{phase_epochs} "
                    f"nominal={nominal_score:.4f} failure_k={failure_score:.4f} mixture={mixture_score:.4f} "
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
                setup=setup_duration,
                sync=sync_duration,
                reset=reset_duration,
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
