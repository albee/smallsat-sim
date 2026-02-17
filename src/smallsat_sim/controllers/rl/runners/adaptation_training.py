import os

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

    subkeys_train = self._take_keys(self.epochs)

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
    self.am_optimizer = nnx.Optimizer(self.am, am_optax)

    num_envs = self.env.num_envs
    history_len = self.env.history_len
    state_action_dim = self.state_action_dim

    for epoch in range(self.epochs):
        self.env.reset_perturbations()
        self.env.apply_random_perturbations(
            key=subkeys_train[epoch],
            fraction_perturbed_envs=0.4,
            perturbation_distribution=jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
        )
        self.env.apply_random_disturbance(
            key=subkeys_train[epoch],
            fraction_disturbed_envs=0.1,
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
            del _step, residuals  # unused
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
        self.agent.key = rollout_result.final_rng

        step_outputs = rollout_result.step_outputs
        actions_traj = rollout_result.actions
        actual_wrench = step_outputs.actual_wrench
        next_obs = step_outputs.next_obs
        history_mask = rollout_result.aux.astype(bool)
        done_masks = rollout_result.done_masks

        state_action_data = jnp.concatenate(
            [step_outputs.prev_states, actions_traj], axis=2
        )
        extrinsics = actual_wrench

        state_action_data, extrinsics = self._build_sliding_windows(
            state_action_data,
            extrinsics,
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
            step_idx, prev_states_step, actions_step, actual_step, done_step = (
                scan_inputs
            )

            combined = jnp.concatenate([prev_states_step, actions_step], axis=1)
            history = jnp.roll(history, shift=-1, axis=1)
            history = history.at[:, -1, :].set(combined)
            counts = jnp.minimum(counts + 1, history_len)
            history_full = counts >= history_len

            all_full = jnp.all(history_full)
            extrinsic_est = jax.lax.cond(
                all_full,
                lambda _: self.adaptation_module(history),
                lambda _: jnp.zeros_like(actual_step),
                operand=None,
            )
            extrinsic_err = jax.lax.cond(
                all_full,
                lambda _: calc_extrinsic_error(extrinsic_est, actual_step),
                lambda _: jnp.zeros((num_envs,), dtype=actual_step.dtype),
                operand=None,
            )
            extrinsic_mean = extrinsic_err.mean()

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
            extrinsics,
            key=split_key,
            shuffle=False,
        )

        train_loss_sum = 0.0
        train_loss_count = 0
        val_loss_sum = 0.0
        val_loss_count = 0
        last_val_loss = None
        am_train_loss_value = 0.0

        for nn_epoch in range(self.rl_cfg.am_epochs):
            am_train_loss, grads = self.jitted_batched_am_loss_and_grad(
                self.am,
                X_train,
                y_train,
            )
            am_train_loss_value = float(am_train_loss)
            train_loss_sum += am_train_loss_value
            train_loss_count += 1
            self.am_optimizer.update(grads)

            if nn_epoch % 10 == 0:
                am_val_loss, _ = self.jitted_batched_am_loss_and_grad(
                    self.am, X_val, y_val
                )
                am_val_loss_value = float(am_val_loss)
                val_loss_sum += am_val_loss_value
                val_loss_count += 1
                last_val_loss = am_val_loss_value

        mean_am_train_loss = (
            train_loss_sum / train_loss_count if train_loss_count > 0 else 0.0
        )
        mean_am_val_loss = (
            val_loss_sum / val_loss_count if val_loss_count > 0 else 0.0
        )

        if self.env.use_wandb:
            wandb.log(
                {
                    "am_train_loss_last": am_train_loss_value,
                    "am_train_loss_mean": mean_am_train_loss,
                    "am_val_loss_last": (
                        last_val_loss if last_val_loss is not None else float("nan")
                    ),
                    "am_val_loss_mean": (
                        mean_am_val_loss if val_loss_count > 0 else float("nan")
                    ),
                }
            )

        if self.agent.has_logger:
            self.env.logger.log(
                self.env.run_id,
                float(self.env.mjx_batch.time[0]),
                step=int(epoch),
                run_name=self.env.run_name,
                stage="am_training",
                am_train_loss_last=am_train_loss_value,
                am_train_loss_mean=mean_am_train_loss,
                am_val_loss_last=(
                    last_val_loss if last_val_loss is not None else 0.0
                ),
                am_val_loss_mean=mean_am_val_loss,
                mean_lateral_error=float(tracking_vals.mean()),
                mean_angle_error=float(angle_vals.mean()),
                mean_extrinsic_error=float(extrinsic_vals.mean()),
            )

        save_adaptation_module(
            self.am, self.ckpt_dir, self.adaptation_module_file_name
        )
