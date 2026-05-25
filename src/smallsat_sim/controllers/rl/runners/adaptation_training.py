import os
import time

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb

from smallsat_sim.controllers.rl.runners.rollout import (
    AdaptationRolloutExtra,
    FunctionalRolloutCallbacks,
    prepare_policy_input_with_residuals,
    residuals_from_wrench_delta,
    run_functional_rollout,
    update_history_buffer,
)
from smallsat_sim.controllers.rl.runners.adaptive_context import (
    build_adaptation_query,
    task_authority_targets,
)
from smallsat_sim.controllers.rl.runners.curriculum import (
    build_authority_regime_curriculum,
    build_difficulty_curriculum,
    build_failure_curriculum,
    uniform_failure_distribution,
)
from smallsat_sim.controllers.rl.runners.failure_scenarios import (
    SPLIT_TRAIN,
    apply_sampled_failure_scenario_split,
    build_failure_scenario_table,
    precompute_scenario_gp_samples,
)
from smallsat_sim.controllers.rl.runners.runner_utils import (
    load_trained_modules,
    save_adaptation_module,
)
from smallsat_sim.envs.vec_env import _compute_state_features
from smallsat_sim.utils.helpers_jax import (
    calc_attitude_error,
    calc_extrinsic_error,
    calc_lateral_tracking_error,
    train_val_split,
)


def train_adaptation_module_on_policy_runner(self) -> None:
    """
    Train adaptation module to predict extrinsics from the history of states and actions with
    on-policy data (RMA approach).
    NOTE: use_adaptive_approach must be set to True in the environment config.
    """
    if self.env.use_adaptive_approach is False:
        return

    file_path = os.path.join(self.ckpt_dir, self.adaptation_module_file_name)
    if os.path.isfile(file_path):
        return

    file_path = os.path.join(self.ckpt_dir, self.training_state_file_name)
    if os.path.isfile(file_path):
        restored_state = load_trained_modules(
            self.ckpt_dir, self.training_state_file_name
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
        raise Exception(
            "The base policy must be trained before the adaptation module."
        )

    print("Training adaptation module...\n")

    self.env.reset()
    self.env.reset_perturbations()
    if hasattr(self.env, "reset_disturbances"):
        self.env.reset_disturbances()

    subkeys_train = self._take_keys(self.epochs)

    cfg = self.env.env_cfg.control.RL
    am_lr = self.env.env_cfg.control.RL.am_lr
    am_weight_decay = self.env.env_cfg.control.RL.am_weight_decay
    grad_clip_norm = max(float(self.env.env_cfg.control.RL.am_grad_clip_norm), 1e-6)
    am_optax = optax.chain(
        optax.clip_by_global_norm(grad_clip_norm),
        optax.adamw(
            learning_rate=am_lr,
            eps=1e-8,
            weight_decay=am_weight_decay,
        ),
    )
    am_checkpoint_interval = int(getattr(cfg, "am_checkpoint_interval", 10))
    am_checkpoint_interval = max(1, am_checkpoint_interval)
    self.am_optimizer = nnx.Optimizer(self.am, am_optax)

    num_envs = self.env.num_envs
    history_len = self.env.history_len
    state_action_dim = self.state_action_dim
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
    curriculum_mode = str(getattr(cfg, "failure_curriculum_mode", "type"))
    use_controllable_failure_scenarios = bool(
        getattr(cfg, "use_controllable_failure_scenarios", False)
    )
    failure_scenario_table = None
    if use_controllable_failure_scenarios:
        failure_scenario_table = build_failure_scenario_table(
            self.env._thruster_mixer_T,
            self.agent.actor.act_low,
            self.agent.actor.act_high,
            max_faults=int(getattr(cfg, "failure_scenario_max_faults", 2)),
            min_rank=int(getattr(cfg, "failure_scenario_min_rank", 6)),
            stress_quantile=float(
                getattr(cfg, "failure_scenario_stress_quantile", 0.9)
            ),
            mild_effectiveness=float(
                getattr(cfg, "failure_scenario_mild_effectiveness", 0.5)
            ),
            sparse_active_thrusters=getattr(
                cfg, "failure_scenario_sparse_active_thrusters", None
            ),
            sparse_max_active_sets=int(
                getattr(cfg, "failure_scenario_sparse_max_active_sets", 32)
            ),
        )
        precompute_scenario_gp_samples(self.env, self._take_keys())

    if (
        curriculum_mode == "authority"
        and use_controllable_failure_scenarios
        and failure_scenario_table is not None
    ):
        curriculum_phases, curriculum_total_epochs = build_authority_regime_curriculum(
            train_with_failures=bool(self.env.train_with_failures),
            fallback_epochs=int(self.epochs),
            nominal_epochs=int(cfg.curriculum_nominal_epochs),
            phase_epochs=int(cfg.curriculum_phase_epochs),
            failure_fraction=float(cfg.curriculum_failure_fraction),
            disturbance_fraction=float(cfg.curriculum_disturbance_fraction),
        )
    elif (
        curriculum_mode == "difficulty"
        and use_controllable_failure_scenarios
        and failure_scenario_table is not None
    ):
        curriculum_phases, curriculum_total_epochs = build_difficulty_curriculum(
            train_with_failures=bool(self.env.train_with_failures),
            fallback_epochs=int(self.epochs),
            nominal_epochs=int(cfg.curriculum_nominal_epochs),
            phase_epochs=int(cfg.curriculum_phase_epochs),
            failure_fraction=float(cfg.curriculum_failure_fraction),
            disturbance_fraction=float(cfg.curriculum_disturbance_fraction),
        )
    else:
        curriculum_phases, curriculum_total_epochs = build_failure_curriculum(
            train_with_failures=bool(self.env.train_with_failures),
            fallback_epochs=int(self.epochs),
            nominal_epochs=int(cfg.curriculum_nominal_epochs),
            phase_epochs=int(cfg.curriculum_phase_epochs),
            failure_fraction=float(cfg.curriculum_failure_fraction),
            disturbance_fraction=float(cfg.curriculum_disturbance_fraction),
        )
    phase_end_epochs = []
    running_epoch = 0
    for phase in curriculum_phases:
        running_epoch += int(phase["epochs"])
        phase_end_epochs.append(running_epoch)

    def _sample_start_time(key, low: float, high: float) -> float:
        if high <= low:
            return low
        return float(jax.random.uniform(key, (), minval=low, maxval=high))

    def _phase_for_am_epoch(epoch: int) -> dict:
        # Project AM collection epochs onto the policy curriculum timeline.
        scaled_epoch = int(epoch * curriculum_total_epochs / max(int(self.epochs), 1))
        scaled_epoch = min(max(scaled_epoch, 0), curriculum_total_epochs - 1)
        for phase, end_epoch in zip(curriculum_phases, phase_end_epochs, strict=True):
            if scaled_epoch < end_epoch:
                return phase
        return curriculum_phases[-1]

    for epoch in range(self.epochs):
        epoch_start_time = time.perf_counter()
        self.env.reset_perturbations()
        if hasattr(self.env, "reset_disturbances"):
            self.env.reset_disturbances()
        phase = _phase_for_am_epoch(epoch)
        phase_failure_fraction = float(phase["failure_fraction"])
        phase_disturbance_fraction = float(phase["disturbance_fraction"])
        phase_difficulty_bin = phase.get("difficulty_bin")
        phase_authority_regime = phase.get("authority_regime")
        phase_task_feasibility_regime = phase.get("task_feasibility_regime")
        phase_distribution = uniform_failure_distribution(
            list(phase["active_failures"])
        )
        (
            perturb_key,
            disturb_key,
            failure_onset_key,
            disturbance_onset_key,
        ) = jax.random.split(subkeys_train[epoch], 4)
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
        if phase_failure_fraction > 0.0:
            if (
                use_controllable_failure_scenarios
                and failure_scenario_table is not None
                and curriculum_mode in ("authority", "difficulty")
            ):
                apply_sampled_failure_scenario_split(
                    self.env,
                    key=perturb_key,
                    table=failure_scenario_table,
                    split_id=SPLIT_TRAIN,
                    fraction_perturbed_envs=phase_failure_fraction,
                    start_time=failure_start_time,
                    difficulty_bin=phase_difficulty_bin,
                    authority_regime=phase_authority_regime,
                    task_feasibility_regime=phase_task_feasibility_regime,
                )
            else:
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

        step_config = self.env.build_step_config(max_episode_len=self.max_ep_len)
        vec_state = self.env.state_struct
        actor_state, critic_state = self.agent.actor_critic_state()

        def _prepare_policy_input(_step, states, residuals, carry_extra):
            return prepare_policy_input_with_residuals(
                _step, states, residuals, carry_extra
            )

        def _sample_policy(_step, policy_input, rng_key, carry_extra):
            del _step  # unused
            actions = self.agent.get_control_input("am_training", policy_input)
            values = jnp.zeros((num_envs,), dtype=jnp.float32)
            logp = jnp.zeros((num_envs,), dtype=jnp.float32)
            return actions, values, logp, rng_key, carry_extra

        def _post_step(
            _step, step_output, actions, residuals, reset_flag, carry_extra
        ):
            del _step
            _, _, history_full, new_extra = update_history_buffer(
                carry_extra=carry_extra,
                prev_states=step_output.prev_states,
                actions=actions,
                reset_flag=reset_flag,
                history_len=history_len,
            )
            residuals_next = residuals_from_wrench_delta(
                step_output=step_output,
                residuals=residuals,
                use_adaptive_approach=True,
                adaptive_context_mode=self.env.adaptive_context_mode,
                thruster_mixer_T=self.env._thruster_mixer_T,
            )
            return residuals_next, history_full, new_extra

        def _bootstrap_value(step_idx, env_state, residuals, rng_key, carry_extra):
            rng_key, value_key = jax.random.split(rng_key)
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

        extra = AdaptationRolloutExtra(
            history=jnp.zeros((num_envs, history_len, state_action_dim)),
            counts=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

        rollout_start_time = time.perf_counter()
        rollout_result = run_functional_rollout(
            step_config=step_config,
            initial_state=vec_state,
            initial_residuals=jnp.zeros((num_envs, self.env.res_dim)),
            rng=self.agent.key,
            num_steps=self.steps_per_epoch,
            reference_waypoint=self.reference_point,
            callbacks=FunctionalRolloutCallbacks(
                prepare_policy_input=_prepare_policy_input,
                sample_policy=_sample_policy,
                post_step=_post_step,
                bootstrap_value=_bootstrap_value,
            ),
            extra=extra,
        )

        jax.block_until_ready(rollout_result.actions)
        rollout_duration = time.perf_counter() - rollout_start_time
        self.agent.key = rollout_result.final_rng

        target_start_time = time.perf_counter()
        step_outputs = rollout_result.step_outputs
        actions_traj = rollout_result.actions
        next_obs = step_outputs.next_obs
        history_mask = rollout_result.aux.astype(bool)
        done_masks = rollout_result.done_masks

        state_action_data = jnp.concatenate(
            [step_outputs.prev_states, actions_traj], axis=2
        )
        query_data = build_adaptation_query(
            states=step_outputs.prev_states.reshape(-1, self.env.obs_dim),
            desired_wrench=step_outputs.desired_wrench.reshape(-1, 6),
            use_task_conditioned_am=self.env.use_task_conditioned_am,
        ).reshape(self.steps_per_epoch, num_envs, self.env.am_query_dim)
        # Train the AM to reproduce the exact context that phase-1 PPO consumed,
        # including stateful terms such as the persistent bias estimate.
        extrinsics = rollout_result.residuals
        delta_states = step_outputs.next_states - step_outputs.prev_states
        tracking_targets = calc_lateral_tracking_error(
            next_obs.reshape(-1, next_obs.shape[-1]), self.planner
        ).reshape(self.steps_per_epoch, num_envs, 1)
        authority_targets = task_authority_targets(
            commanded_ctrl=step_outputs.commanded_ctrl.reshape(-1, self.env.act_dim),
            applied_ctrl=step_outputs.applied_ctrl.reshape(-1, self.env.act_dim),
            actual_wrench=step_outputs.actual_wrench.reshape(-1, 6),
            desired_wrench=step_outputs.desired_wrench.reshape(-1, 6),
        ).reshape(self.steps_per_epoch, num_envs, 3)
        am_targets = jnp.concatenate(
            [extrinsics, query_data, delta_states, tracking_targets, authority_targets],
            axis=2,
        )
        actual_wrench = step_outputs.actual_wrench

        state_action_data, am_targets = self._build_sliding_windows(
            state_action_data,
            am_targets,
            history_mask,
            history_len,
        )
        if state_action_data.shape[0] == 0:
            print(
                "Skipping adaptation module update: insufficient full-history samples."
            )
            continue

        obs_flat = next_obs.reshape(-1, next_obs.shape[-1])
        tracking_vals = calc_lateral_tracking_error(
            obs_flat, self.planner
        ).reshape(self.steps_per_epoch, num_envs)
        angle_vals = jnp.degrees(calc_attitude_error(obs_flat)).reshape(
            self.steps_per_epoch, num_envs
        )
        tracking_vals = tracking_vals.mean(axis=1)
        angle_vals = angle_vals.mean(axis=1)

        def _history_scan(carry, scan_inputs):
            history, counts = carry
            step_idx, prev_states_step, actions_step, actual_step, desired_step, done_step = (
                scan_inputs
            )

            combined = jnp.concatenate([prev_states_step, actions_step], axis=1)
            history = jnp.roll(history, shift=-1, axis=1)
            history = history.at[:, -1, :].set(combined)
            counts = jnp.minimum(counts + 1, history_len)
            history_full = counts >= history_len

            query = build_adaptation_query(
                states=prev_states_step,
                desired_wrench=desired_step,
                use_task_conditioned_am=self.env.use_task_conditioned_am,
            )
            extrinsic_est_raw = self.adaptation_module(history, query)
            extrinsic_est = jnp.where(
                history_full[:, None],
                extrinsic_est_raw,
                jnp.zeros_like(extrinsic_est_raw),
            )
            estimated_wrench = extrinsic_est[:, :6] + desired_step
            extrinsic_err = calc_extrinsic_error(estimated_wrench, actual_step)
            mask_f = history_full.astype(extrinsic_err.dtype)
            extrinsic_mean = (extrinsic_err * mask_f).sum() / jnp.maximum(
                mask_f.sum(), 1.0
            )

            reset_flag = jnp.logical_and(
                done_step,
                step_idx != (self.steps_per_epoch - 1),
            )
            history = jnp.where(
                reset_flag[:, None, None],
                jnp.zeros_like(history),
                history,
            )
            counts = jnp.where(reset_flag, jnp.zeros_like(counts), counts)

            return (history, counts), extrinsic_mean

        scan_inputs = (
            jnp.arange(self.steps_per_epoch, dtype=jnp.int32),
            step_outputs.prev_states,
            actions_traj,
            actual_wrench,
            step_outputs.desired_wrench,
            done_masks,
        )
        init_history = jnp.zeros((num_envs, history_len, state_action_dim))
        init_counts = jnp.zeros((num_envs,), dtype=jnp.int32)
        (_, _), extrinsic_vals = jax.lax.scan(
            _history_scan, (init_history, init_counts), scan_inputs
        )

        split_key = self._take_keys()
        (
            X_train,
            y_train,
            X_val,
            y_val,
            _,
        ) = train_val_split(
            state_action_data,
            am_targets,
            key=split_key,
            shuffle=False,
        )
        target_duration = time.perf_counter() - target_start_time

        update_start_time = time.perf_counter()
        train_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        train_loss_count = 0
        val_loss_sum = jnp.array(0.0, dtype=jnp.float32)
        val_loss_count = 0
        last_val_loss = None
        am_train_loss_last = jnp.array(0.0, dtype=jnp.float32)

        for nn_epoch in range(self.rl_cfg.am_epochs):
            am_train_loss, grads = self.jitted_batched_am_loss_and_grad(
                self.am,
                X_train,
                y_train,
            )
            am_train_loss_last = am_train_loss
            train_loss_sum = train_loss_sum + am_train_loss
            train_loss_count += 1
            self.am_optimizer.update(grads)

            if nn_epoch % 10 == 0:
                am_val_loss = self.jitted_batched_am_loss_components(
                    self.am, X_val, y_val
                )[-1]
                val_loss_sum = val_loss_sum + am_val_loss
                val_loss_count += 1
                last_val_loss = am_val_loss

        mean_am_train_loss_arr = (
            train_loss_sum / train_loss_count
            if train_loss_count > 0
            else jnp.array(0.0, dtype=jnp.float32)
        )
        mean_am_val_loss_arr = (
            val_loss_sum / val_loss_count
            if val_loss_count > 0
            else jnp.array(0.0, dtype=jnp.float32)
        )
        jax.block_until_ready(am_train_loss_last)
        update_duration = time.perf_counter() - update_start_time

        metrics_start_time = time.perf_counter()
        (
            am_context_loss,
            am_delta_loss,
            am_tracking_loss,
            am_authority_loss,
            am_kl_loss,
            am_total_loss,
        ) = self.jitted_batched_am_loss_components(self.am, X_train, y_train)
        (
            am_val_context_loss,
            am_val_delta_loss,
            am_val_tracking_loss,
            am_val_authority_loss,
            am_val_kl_loss,
            am_val_total_loss,
        ) = self.jitted_batched_am_loss_components(self.am, X_val, y_val)
        am_context_loss = float(am_context_loss)
        am_delta_loss = float(am_delta_loss)
        am_tracking_loss = float(am_tracking_loss)
        am_authority_loss = float(am_authority_loss)
        am_kl_loss = float(am_kl_loss)
        am_total_loss = float(am_total_loss)
        am_val_context_loss = float(am_val_context_loss)
        am_val_delta_loss = float(am_val_delta_loss)
        am_val_tracking_loss = float(am_val_tracking_loss)
        am_val_authority_loss = float(am_val_authority_loss)
        am_val_kl_loss = float(am_val_kl_loss)
        am_val_total_loss = float(am_val_total_loss)
        am_train_loss_value = float(am_train_loss_last)
        mean_am_train_loss = float(mean_am_train_loss_arr)
        last_val_loss_value = (
            float(last_val_loss) if last_val_loss is not None else None
        )
        mean_am_val_loss = float(mean_am_val_loss_arr)
        attention_metrics = {}
        if self.env.am_architecture == "transformer_cross_attention":
            query_dim = int(self.env.am_query_dim)
            if query_dim > 0 and X_val.shape[0] > 0:
                target_all_val = y_val[:, -1, :]
                query_val = target_all_val[
                    :, self.env.res_dim : self.env.res_dim + query_dim
                ]
                _, attention = jax.vmap(
                    lambda hist, query_i: self.am(
                        hist,
                        query_i,
                        return_attention=True,
                    )
                )(X_val, query_val)
                attention_mean = attention.mean(axis=(0, 1))
                time_axis = jnp.linspace(
                    0.0,
                    1.0,
                    attention_mean.shape[0],
                    dtype=attention_mean.dtype,
                )
                attention_entropy = -jnp.sum(
                    attention_mean * jnp.log(attention_mean + 1e-8)
                )
                max_entropy = jnp.log(jnp.asarray(attention_mean.shape[0]))
                attention_metrics = {
                    "am_attention/entropy": float(attention_entropy),
                    "am_attention/normalized_entropy": float(
                        attention_entropy / (max_entropy + 1e-8)
                    ),
                    "am_attention/recency_center": float(
                        jnp.sum(attention_mean * time_axis)
                    ),
                    "am_attention/latest_token_weight": float(attention_mean[-1]),
                    "am_attention/oldest_token_weight": float(attention_mean[0]),
                }
        metrics_duration = time.perf_counter() - metrics_start_time

        logging_start_time = time.perf_counter()
        if self.env.use_wandb:
            wandb_payload = {
                "am_collection_phase": phase["name"],
                "am_collection_failure_fraction": phase_failure_fraction,
                "am_collection_disturbance_fraction": phase_disturbance_fraction,
                "am_train_loss_last": am_train_loss_value,
                "am_train_loss_mean": mean_am_train_loss,
                "am_context_loss": am_context_loss,
                "am_delta_loss": am_delta_loss,
                "am_tracking_loss": am_tracking_loss,
                "am_authority_loss": am_authority_loss,
                "am_kl_loss": am_kl_loss,
                "am_total_loss": am_total_loss,
                "am_val_loss_last": (
                    last_val_loss_value
                    if last_val_loss_value is not None
                    else float("nan")
                ),
                "am_val_loss_mean": (
                    mean_am_val_loss if val_loss_count > 0 else float("nan")
                ),
                "am_val_context_loss": am_val_context_loss,
                "am_val_delta_loss": am_val_delta_loss,
                "am_val_tracking_loss": am_val_tracking_loss,
                "am_val_authority_loss": am_val_authority_loss,
                "am_val_kl_loss": am_val_kl_loss,
                "am_val_total_loss": am_val_total_loss,
            }
            wandb_payload.update(attention_metrics)
            wandb.log(wandb_payload)

        if self.agent.has_logger:
            logger_attention_metrics = {
                key.replace("/", "_"): value for key, value in attention_metrics.items()
            }
            self.env.logger.log(
                self.env.run_id,
                float(self.env.mjx_batch.time[0]),
                step=int(epoch),
                run_name=self.env.run_name,
                stage="am_training",
                am_collection_phase=phase["name"],
                am_collection_failure_fraction=phase_failure_fraction,
                am_collection_disturbance_fraction=phase_disturbance_fraction,
                am_train_loss_last=am_train_loss_value,
                am_train_loss_mean=mean_am_train_loss,
                am_context_loss=am_context_loss,
                am_delta_loss=am_delta_loss,
                am_tracking_loss=am_tracking_loss,
                am_authority_loss=am_authority_loss,
                am_kl_loss=am_kl_loss,
                am_total_loss=am_total_loss,
                am_val_loss_last=(
                    last_val_loss_value if last_val_loss_value is not None else 0.0
                ),
                am_val_loss_mean=mean_am_val_loss,
                am_val_context_loss=am_val_context_loss,
                am_val_delta_loss=am_val_delta_loss,
                am_val_tracking_loss=am_val_tracking_loss,
                am_val_authority_loss=am_val_authority_loss,
                am_val_kl_loss=am_val_kl_loss,
                am_val_total_loss=am_val_total_loss,
                mean_lateral_error=float(tracking_vals.mean()),
                mean_angle_error=float(angle_vals.mean()),
                mean_extrinsic_error=float(extrinsic_vals.mean()),
                **logger_attention_metrics,
            )
        logging_duration = time.perf_counter() - logging_start_time

        save_start_time = time.perf_counter()
        should_save_checkpoint = (
            (epoch + 1) % am_checkpoint_interval == 0
            or epoch == self.epochs - 1
        )
        if should_save_checkpoint:
            save_adaptation_module(
                self.am, self.ckpt_dir, self.adaptation_module_file_name
            )
        save_duration = time.perf_counter() - save_start_time
        total_duration = time.perf_counter() - epoch_start_time
        print(
            f"[AM Timing] Epoch {epoch + 1}/{self.epochs}: "
            f"rollout {rollout_duration:.2f}s | targets {target_duration:.2f}s | "
            f"update {update_duration:.2f}s | metrics {metrics_duration:.2f}s | "
            f"logging {logging_duration:.2f}s | save {save_duration:.2f}s | "
            f"total {total_duration:.2f}s"
        )
