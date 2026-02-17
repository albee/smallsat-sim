import os
import time
import warnings

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb

from smallsat_sim.controllers.rl.runners.rollout_utils import (
    AdaptationRolloutExtra,
    FunctionalRolloutCallbacks,
    run_functional_rollout,
)
from smallsat_sim.controllers.rl.runners.curriculum import (
    build_failure_curriculum,
    uniform_failure_distribution,
)
from smallsat_sim.envs.vec_env import VecEnv, _compute_state_features
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController
from smallsat_sim.controllers.rl.algorithms.vpg import VPG
from smallsat_sim.controllers.rl.algorithms.ppo import PPO
from smallsat_sim.controllers.rl.modules.am_cnn import CNNAdaptationModule
from smallsat_sim.controllers.rl.modules.am_transformer import (
    TransformerAdaptationModule,
)
from smallsat_sim.controllers.rl.storage.replay_buffer import ReplayBuffer
from smallsat_sim.controllers.rl.runners.runner_utils import (
    save_training_data,
    save_trained_modules,
    save_adaptation_module,
    load_training_data,
    load_trained_modules,
)
from smallsat_sim.utils.helpers_jax import (
    train_val_split,
    mae_loss_fn,
    calc_lateral_tracking_error,
    calc_attitude_error,
    calc_extrinsic_error,
)
from smallsat_sim.utils.wandb_config import setup_wandb


class OnPolicyRunner(object):
    """
    On-policy runner for training and evaluation. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """

    def __init__(self, env: VecEnv, planner: OraclePlannerRL) -> None:
        # Initialize the environment and agent
        self.env = env
        self.planner = planner
        base_key = self.env.next_rng_keys(1)[0]
        self._rng, agent_key = jax.random.split(base_key)
        self.agent = PPO(self.env, planner, rng_key=agent_key)
        self.state_action_dim = self.env.obs_dim + self.env.act_dim
        self.am = self._build_adaptation_module()
        self.reference_point = planner.get_reference(
            self.env.get_obs()
        )  # Is re-used in every episode
        self.pd_ctrl = VectorizedPDController(env, planner)
        self._load_rl_hyperparams()

        # Vectorize adaptation module
        self.adaptation_module = jax.vmap(self.am)
        self.am_kl_weight = self.env.env_cfg.control.RL.am_kl_weight
        self.am_loss_fn = self._select_am_loss_fn()

        # JIT-compile the adaptation module updates
        self.jitted_batched_am_loss_and_grad = nnx.jit(
            nnx.value_and_grad(self.am_loss_fn), static_argnums=()
        )

        self.rl_cfg = self.env.env_cfg.control.RL
        # Optional regression check: compare one functional step against the
        # imperative legacy transition path. This is useful while refactoring
        # but should typically stay disabled during normal training
        self._functional_check_enabled = bool(
            getattr(self.rl_cfg, "verify_functional_rollout", False)
        )
        self._functional_check_ran = False
        self._functional_check_atol = float(
            getattr(self.rl_cfg, "verify_functional_rollout_atol", 1e-4)
        )
        self._functional_check_rtol = float(
            getattr(self.rl_cfg, "verify_functional_rollout_rtol", 1e-3)
        )

        # Path to save the checkpoints
        self.ckpt_dir = "src/smallsat_sim/controllers/rl/checkpoints/"

        # Checkpoint file names
        self._create_checkpoint_file_names()

        # Use Weights and Biases for logging
        if self.env.use_wandb:
            setup_wandb()
            wandb.init(
                project="Astrobee-training",
                name=f"{self.env.run_name}_{self.env.run_id}",
                config={
                    "num_envs": self.env.num_envs,
                    "steps_per_epoch": self.steps_per_epoch,
                    "epochs": self.epochs,
                    "max_ep_len": self.max_ep_len,
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "actor_lr": self.agent.actor_lr,
                    "critic_lr": self.agent.critic_lr,
                    "episode_len": self.episode_len,
                    "n_evals": self.n_evals,
                },
            )

    def _take_keys(self, count: int = 1):
        """
        Consume ``count`` RNG keys from the runner seed.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        splits = jax.random.split(self._rng, count + 1)
        self._rng = splits[0]
        if count == 1:
            return splits[1]
        return splits[1:]

    def pretrain(self, strategy: str = "supervised_learning") -> None:
        """
        Pretrain the actor and critic networks.

        Supported strategy:
        - ``supervised_learning`` (recommended)

        Deprecated strategy:
        - ``rl`` (kept temporarily for backward compatibility)
        """
        # Check if pretraining has already been done
        file_path = os.path.join(self.ckpt_dir, self.pretraining_state_file_name)
        if os.path.isfile(file_path):
            return

        print("Pretraining modules...\n")

        # Generate experience if necessary and load the pretraining data
        self._generate_experience()
        pretraining_data = load_training_data(
            self.ckpt_dir, self.pretraining_data_file_name
        )

        # Load the data
        obs = pretraining_data["obs"].reshape(-1, self.env.obs_dim)
        act = pretraining_data["act"].reshape(-1, self.env.act_dim)
        ret = pretraining_data["ret"].reshape(-1)
        tdres = pretraining_data["tdres"].reshape(-1)
        logp = pretraining_data["logp"].reshape(-1)
        if self.env.use_adaptive_approach is True:
            residuals = pretraining_data["residuals"].reshape(-1, self.env.res_dim)
        else:
            residuals = jnp.empty((self.steps_per_epoch * self.env.num_envs, 0))

        # Clip the actions to the highest upper bound on the force range of the thrusters
        act_clipped = jnp.where(act > 0.6, 0.6, act)

        # Pretrain the policy network
        if strategy == "supervised_learning":
            # Optimizer to pretrain the policy network
            actor_optimizer = nnx.Optimizer(
                self.agent.actor, optax.adam(learning_rate=1e-2, eps=1e-5)
            )

            # Split into training and validation sets
            split_key = self._take_keys()
            (
                X_train,
                y_train,
                X_val,
                y_val,
                _,
            ) = train_val_split(
                jnp.concatenate([obs, residuals], axis=1),
                act_clipped,
                key=split_key,
            )

            num_epochs = 80
            num_batches = 256
            batch_size = int(jnp.ceil(obs.shape[0] / num_batches))
            num_train_samples = X_train.shape[0]
            num_val_samples = X_val.shape[0]

            # Create PRNG keys
            subkeys_pretrain = self._take_keys(num_epochs)

            # Training loop
            for epoch in range(num_epochs):
                # Shuffle the training data
                indices = jax.random.permutation(
                    subkeys_pretrain[epoch], num_train_samples
                )
                X_train = X_train[indices]
                y_train = y_train[indices]

                epoch_train_loss_sum = jnp.array(0.0)
                num_train_batches = 0
                for i in range(0, num_train_samples, batch_size):
                    batch_X = X_train[i : i + batch_size]
                    batch_y = y_train[i : i + batch_size]

                    # Train network
                    actor_loss, grads = nnx.value_and_grad(mae_loss_fn)(
                        self.agent.actor, batch_X, batch_y, subkeys_pretrain[epoch]
                    )
                    actor_optimizer.update(grads)
                    epoch_train_loss_sum = epoch_train_loss_sum + actor_loss
                    num_train_batches += 1

                avg_train_loss = epoch_train_loss_sum / max(num_train_batches, 1)
                print(
                    f"Epoch: {epoch+1:2} avg. actor training loss: {avg_train_loss}\n"
                )

                epoch_val_loss_sum = jnp.array(0.0)
                num_val_batches = 0
                for i in range(0, num_val_samples, batch_size):
                    batch_X_val = X_val[i : i + batch_size]
                    batch_y_val = y_val[i : i + batch_size]

                    # Validate
                    actor_val_loss, _ = nnx.value_and_grad(mae_loss_fn)(
                        self.agent.actor,
                        batch_X_val,
                        batch_y_val,
                        subkeys_pretrain[epoch],
                    )
                    epoch_val_loss_sum = epoch_val_loss_sum + actor_val_loss
                    num_val_batches += 1

                avg_val_loss = epoch_val_loss_sum / max(num_val_batches, 1)
                print(
                    f"Epoch: {epoch+1:2} avg. actor evaluation loss: {avg_val_loss}\n"
                )
                if self.env.use_wandb:
                    wandb.log(
                        {
                            "training_loss": float(avg_train_loss),
                            "validation_loss": float(avg_val_loss),
                        }
                    )
        elif strategy == "rl":
            warnings.warn(
                "Pretraining strategy 'rl' is deprecated and will be removed in a future release. "
                "Use strategy='supervised_learning' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            self.agent.update_policy_gradient(
                self._take_keys(), obs, act_clipped, tdres, logp
            )
        else:
            raise Exception(
                "Unknown pretraining strategy. Use strategy='supervised_learning'. "
                "Strategy='rl' is deprecated."
            )

        # Pretrain the base network
        self.agent.update_value_function(
            self._take_keys(),
            jnp.concatenate([obs, residuals], axis=1),
            ret,
        )

        # Only pretrain the actor mean, reset log_std
        self.agent.actor.log_std = nnx.Param(-0.5 * jnp.ones(self.env.act_dim))

        # Save the trained actor and critic network weights
        save_trained_modules(
            self.agent, self.ckpt_dir, self.pretraining_state_file_name
        )

    def learn(self) -> None:
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
                    del _step, carry_extra  # unused
                    if residuals.shape[-1]:
                        return jnp.concatenate([states, residuals], axis=1), None
                    return states, None

                def _sample_eval_policy(_step, policy_input, rng_key, carry_extra):
                    del _step, carry_extra  # unused
                    actions = self.agent.get_control_input("evaluation", policy_input)
                    zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
                    return actions, zeros, zeros, rng_key, None

                def _post_eval_step(
                    _step, step_output, actions, residuals, reset_flag, carry_extra
                ):
                    del _step, actions, reset_flag, carry_extra  # unused
                    if self.env.use_adaptive_approach:
                        residuals_next = (
                            step_output.actual_wrench - step_output.desired_wrench
                        )
                    else:
                        residuals_next = residuals
                    return residuals_next, None, None

                def _bootstrap_eval(_step, env_state, residuals, rng_key, carry_extra):
                    del _step, env_state, residuals, carry_extra  # unused
                    zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
                    return zeros, rng_key, None

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
                perturb_key, disturb_key, actor_key, eval_key = jax.random.split(
                    epoch_key, 4
                )

                # Apply failures/disturbances for this phase with fixed proportions
                if self.env.train_with_failures:
                    self.env.reset_perturbations()  # avoid accumulating failures across epochs
                    if phase_failure_fraction > 0.0:
                        self.env.apply_random_perturbations(
                            key=perturb_key,
                            fraction_perturbed_envs=phase_failure_fraction,
                            perturbation_distribution=phase_distribution,
                        )
                    if phase_disturbance_fraction > 0.0:
                        self.env.apply_random_disturbance(
                            key=disturb_key,
                            fraction_disturbed_envs=phase_disturbance_fraction,
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
                    del _step  # unused
                    if residuals.shape[-1]:
                        return jnp.concatenate([states, residuals], axis=1), carry_extra
                    return states, carry_extra

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
                    if self.env.use_adaptive_approach:
                        residuals_next = (
                            step_output.actual_wrench - step_output.desired_wrench
                        )
                    else:
                        residuals_next = residuals
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

                terminals_any_epoch = jnp.any(step_outputs.terminals, axis=0)
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
                other_rollout_time = max(
                    0.0, rollout_duration - (scan_time + buffer_time)
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

                adv_mean_f = float(tdres.mean())
                adv_std_f = float(tdres.std())
                value_mean_f = float(vals.mean())
                value_std_f = float(vals.std())
                return_mean_f = float(returns.mean())
                return_std_f = float(returns.std())
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
                    wandb_reward_payload = {
                        f"reward_components/{metric_name}": float(metric_value)
                        for metric_name, metric_value in reward_component_means.items()
                    }
                    wandb_reward_scale_payload = {
                        f"reward_scale/{metric_name}": metric_value
                        for metric_name, metric_value in reward_scale_metrics.items()
                    }
                    wandb.log(
                        {
                            "curriculum/phase": phase_name,
                            "curriculum/phase_epoch": phase_epoch + 1,
                            "curriculum/global_epoch": global_epoch,
                            "curriculum/critic_grad_scale": critic_grad_scale,
                            "training_health/actor_loss_mean": actor_loss_mean_f,
                            "training_health/critic_loss_mean": critic_loss_mean_f,
                            "training_health/true_kl_mean": kl_mean_f,
                            "training_health/clip_fraction": clip_frac_f,
                            "training_health/explained_variance": explained_var_f,
                            "policy_stats/mean_std": float(
                                jnp.exp(self.agent.actor.log_std.value).mean()
                            ),
                            "policy_stats/mean_log_std": float(
                                self.agent.actor.log_std.value.mean()
                            ),
                            "task_performance/mean_episodic_returns": float(
                                mean_ep_return_epoch
                            ),
                            "objective/train_mean_episodic_return": float(
                                mean_ep_return_epoch
                            ),
                            "task_performance/success_env_count": float(
                                success_env_count_epoch
                            ),
                            "task_performance/success_rate": float(
                                success_rate_epoch
                            ),
                            "task_performance/terminated_step_count": float(
                                terminated_step_count_epoch
                            ),
                            "task_performance/success_termination_step_count": float(
                                success_termination_step_count_epoch
                            ),
                            "task_performance/failure_termination_step_count": float(
                                failure_termination_step_count_epoch
                            ),
                            "task_performance/success_termination_env_rate": float(
                                success_termination_env_rate_epoch
                            ),
                            "task_performance/failure_termination_env_rate": float(
                                failure_termination_env_rate_epoch
                            ),
                            "task_performance/terminal_envs_at_end": float(
                                terminal_envs_at_end_epoch
                            ),
                            "task_performance/terminal_env_rate_at_end": float(
                                terminal_env_rate_at_end_epoch
                            ),
                            "task_performance/mean_lateral_error": float(
                                tracking_error_epoch
                            ),
                            "task_performance/mean_angle_error": float(
                                angle_error_epoch
                            ),
                            "task_performance/mean_final_position_error": float(
                                final_pos_error_epoch
                            ),
                            **wandb_reward_payload,
                            **wandb_reward_scale_payload,
                        },
                        step=global_epoch,
                    )

                # Log key RL metrics
                if self.agent.has_logger:
                    self.env.logger.log(
                        self.env.run_id,
                        float(self.env.mjx_batch.time[0]),
                        step=int(global_epoch),
                        run_name=self.env.run_name,
                        stage="policy_training",
                        phase=phase_name,
                        phase_epoch=int(phase_epoch + 1),
                        critic_grad_scale=float(critic_grad_scale),
                        actor_loss_mean=actor_loss_mean_f,
                        critic_loss_mean=critic_loss_mean_f,
                        true_kl_mean=kl_mean_f,
                        clip_fraction=clip_frac_f,
                        explained_variance=explained_var_f,
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
                        mean_std=float(jnp.exp(self.agent.actor.log_std.value).mean()),
                        mean_lateral_error=float(tracking_error_epoch),
                        mean_angle_error=float(angle_error_epoch),
                        mean_final_position_error=float(final_pos_error_epoch),
                        train_mean_episodic_returns=float(mean_ep_return_epoch),
                        **{
                            f"reward_scale_{k}": float(v)
                            for k, v in reward_scale_metrics.items()
                        },
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
                save_trained_modules(
                    self.agent, self.ckpt_dir, self.training_state_file_name
                )
                save_duration = time.perf_counter() - save_start_time
                epoch_total_duration = time.perf_counter() - epoch_start_time
                print(
                    f"[Timing] Epoch {global_epoch}/{self.epochs} "
                    f"(phase={phase_name} {phase_epoch + 1}/{phase_epochs}): "
                    f"rollout {rollout_duration:.2f}s "
                    f"(scan {scan_time:.2f}s, buffer {buffer_time:.2f}s, other {other_rollout_time:.2f}s) | "
                    f"update {update_duration:.2f}s | "
                    f"logging {logging_duration:.2f}s | "
                    f"eval {eval_duration:.2f}s | "
                    f"save_ckpt {save_duration:.2f}s | "
                    f"total {epoch_total_duration:.2f}s"
                )

        # Restore the best nominal checkpoint at the end of the curriculum
        if best_actor_state is not None and best_critic_state is not None:
            nnx.update(self.agent.actor, best_actor_state)
            nnx.update(self.agent.critic, best_critic_state)
            save_trained_modules(
                self.agent, self.ckpt_dir, self.training_state_file_name
            )

    def train_adaptation_module_on_policy(self) -> None:
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
                del _step  # unused
                if residuals.shape[-1]:
                    return jnp.concatenate([states, residuals], axis=1), carry_extra
                return states, carry_extra

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
                history = carry_extra.history
                counts = carry_extra.counts

                combined = jnp.concatenate([step_output.prev_states, actions], axis=1)
                history = jnp.roll(history, shift=-1, axis=1)
                history = history.at[:, -1, :].set(combined)
                counts = jnp.minimum(counts + 1, history_len)
                reset_mask = reset_flag[:, None, None]
                history = jnp.where(reset_mask, jnp.zeros_like(history), history)
                counts = jnp.where(reset_flag, jnp.zeros_like(counts), counts)

                history_full = counts >= history_len
                residuals_next = step_output.actual_wrench - step_output.desired_wrench
                new_extra = AdaptationRolloutExtra(history=history, counts=counts)
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

    def posttrain(self) -> None:
        """
        Fine-tune the policy on imperfectly estimated residual wrenches (phase 3 in A-RMA).
        NOTE: use_adaptive_approach must be set to True in the environment config.
        """
        raise NotImplementedError("Post-training is not implemented yet.\n")

    def evaluate(self, phase: int = 2) -> None:
        """
        Evaluate the agent.
        If phase == 1, evaluate base policy before training the adaptation module.
        If phase == 2, evaluate base policy after training the adaptation module.
        """
        # Check if trained actor, critic and adaptation modules are available and load them
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
            if self.env.use_adaptive_approach and phase == 2:
                adapt_module_state = load_trained_modules(
                    self.ckpt_dir, self.adaptation_module_file_name
                )
                am_state = adapt_module_state["am_model"]
                if isinstance(am_state, dict):
                    nnx.update(self.am, am_state)
                else:
                    nnx.update(self.am, am_state)
        else:
            raise Exception("Not all necessary modules have been trained yet.\n")

        if phase not in (1, 2):
            raise ValueError("Phase must be either 1 or 2.")

        subkeys_eval = self._take_keys(self.n_evals)
        subkeys_eval = jnp.atleast_2d(subkeys_eval)

        num_envs = self.env.num_envs
        history_len = self.env.history_len
        state_action_dim = self.state_action_dim
        returns = jnp.zeros((num_envs, self.n_evals), dtype=jnp.float32)

        for eval_idx in range(self.n_evals):
            print(f"Testing policy: episode {eval_idx + 1}/{self.n_evals}")
            self.env.reset()
            self.env.reset_perturbations()

            if self.env.train_with_failures and eval_idx >= self.n_evals // 2:
                eval_key = subkeys_eval[eval_idx]
                self.env.apply_random_perturbations(
                    key=eval_key,
                    fraction_perturbed_envs=0.4,
                    perturbation_distribution=jnp.array([0.2, 0.2, 0.2, 0.2, 0.2]),
                )
                self.env.apply_random_disturbance(
                    key=eval_key,
                    fraction_disturbed_envs=0.1,
                )

            step_config = self.env.build_step_config(
                max_episode_len=self.episode_len,
            )
            vec_state = self.env.state_struct

            if self.env.use_adaptive_approach:
                residual_init = jnp.zeros(
                    (num_envs, self.env.res_dim), dtype=jnp.float32
                )
            else:
                residual_init = jnp.zeros((num_envs, 0), dtype=jnp.float32)

            extra = AdaptationRolloutExtra(
                history=jnp.zeros((num_envs, history_len, state_action_dim)),
                counts=jnp.zeros((num_envs,), dtype=jnp.int32),
            )

            def _prepare_policy_input(_step, states, residuals, carry_extra):
                del _step  # unused
                if residuals.shape[-1]:
                    return jnp.concatenate([states, residuals], axis=1), carry_extra
                return states, carry_extra

            def _sample_policy(_step, policy_input, rng_key, carry_extra):
                del _step  # unused
                actions = self.agent.get_control_input("evaluation", policy_input)
                zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
                return actions, zeros, zeros, rng_key, carry_extra

            def _post_step(
                _step, step_output, actions, residuals, reset_flag, carry_extra
            ):
                del _step, residuals  # unused
                history = carry_extra.history
                counts = carry_extra.counts

                combined = jnp.concatenate([step_output.prev_states, actions], axis=1)
                history = jnp.roll(history, shift=-1, axis=1)
                history = history.at[:, -1, :].set(combined)
                counts = jnp.minimum(counts + 1, history_len)
                history = jnp.where(
                    reset_flag[:, None, None],
                    jnp.zeros_like(history),
                    history,
                )
                counts = jnp.where(reset_flag, jnp.zeros_like(counts), counts)

                if self.env.use_adaptive_approach:
                    desired = step_output.desired_wrench
                    if phase == 1:
                        extrinsic = step_output.actual_wrench
                    else:  # phase == 2 guaranteed by earlier check
                        extrinsic = self.adaptation_module(history)
                    residuals_next = extrinsic - desired
                else:
                    residuals_next = jnp.zeros(
                        (num_envs, 0), dtype=step_output.actual_wrench.dtype
                    )

                history_full = counts >= history_len
                new_extra = AdaptationRolloutExtra(history=history, counts=counts)
                return residuals_next, history_full, new_extra

            def _bootstrap_value(_step, env_state, residuals, rng_key, carry_extra):
                del _step, env_state, residuals  # unused
                zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
                return zeros, rng_key, carry_extra

            rollout_result = run_functional_rollout(
                step_config=step_config,
                initial_state=vec_state,
                initial_residuals=residual_init,
                rng=self.agent.key,
                num_steps=self.episode_len,
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

            self.env._state = rollout_result.final_state
            self.env._rng = rollout_result.final_state.rng
            self.env.mjx_batch = rollout_result.final_state.mjx_batch
            self.env.disturbance_states = rollout_result.final_state.disturbance_states
            self.env.perturbation_states = (
                rollout_result.final_state.perturbation_states
            )
            if hasattr(self.env, "disturbances") and self.env.disturbances is not None:
                for obj, snapshot in zip(
                    self.env.disturbances.disturbances,
                    rollout_result.final_state.disturbance_states,
                    strict=True,
                ):
                    obj.state = snapshot
            if (
                hasattr(self.env, "perturbations")
                and self.env.perturbations is not None
            ):
                for obj, snapshot in zip(
                    self.env.perturbations.perturbations,
                    rollout_result.final_state.perturbation_states,
                    strict=True,
                ):
                    obj.state = snapshot
            self.env._refresh_effect_states()

            step_outputs = rollout_result.step_outputs
            rewards = step_outputs.rewards
            done_masks = rollout_result.done_masks
            terminated_masks = rollout_result.terminated_masks
            truncated_masks = rollout_result.truncated_masks
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
            tracking_seq = calc_lateral_tracking_error(flat_obs, self.planner).reshape(
                self.episode_len, num_envs
            )
            angle_seq = jnp.degrees(calc_attitude_error(flat_obs)).reshape(
                self.episode_len, num_envs
            )
            tracking_mean = float((tracking_seq * mask_f).sum() / denom)
            angle_mean = float((angle_seq * mask_f).sum() / denom)

            if self.env.use_adaptive_approach:
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
            if self.env.collect_reward_components:
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
                step_outputs.terminals[last_active_idx, env_ids]
                .astype(jnp.float32)
                .sum()
            )
            terminal_env_rate_at_end = float(
                step_outputs.terminals[last_active_idx, env_ids]
                .astype(jnp.float32)
                .mean()
            )
            ref_pos = jnp.atleast_2d(self.reference_point)[:, :3]
            final_pos_error = float(
                jnp.linalg.norm(final_positions - ref_pos, axis=1).mean()
            )

            if self.agent.has_logger:
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    step=int(eval_idx),
                    run_name=self.env.run_name,
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

    def _generate_experience(self) -> None:
        """
        Roll out an episode where the actions are computed from a PD controller
        that serves as pretraining data.

        Note:
        This method intentionally uses the legacy imperative
        ``self.env.transition(...)`` pipeline. Keeping it in place provides a
        stable reference path for pretraining and regression comparisons while
        the functional rollout refactor matures.
        """
        # Check if pretraining data already exists
        file_path = os.path.join(self.ckpt_dir, self.pretraining_data_file_name)
        if os.path.isfile(file_path):
            return

        print("Gather training data using the PD controller...\n")

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

        # Initialize the environment
        self.env.reset()
        states, ep_ret, ep_len = (
            self.env.get_states(self.reference_point),
            jnp.zeros(self.env.num_envs),
            0,
        )

        if self.env.use_adaptive_approach is True:
            res = jnp.zeros((self.env.num_envs, self.env.res_dim))
        else:
            res = jnp.empty((self.env.num_envs, 0))

        # Epoch variables
        ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))

        # Main training loop
        for t in range(self.steps_per_epoch):
            # Get value estimates from the agent
            _, v, logp = self.agent.act(
                jnp.concatenate([states, res], axis=1)
            )  # Use un-normalized states
            self.env.obs = self.env.get_obs()

            # Compute actions from the PD controller
            a = self.pd_ctrl.get_control_input(self.env)

            # Perform environment transition
            r, terminal = self.env.transition(
                a, jnp.concatenate([states, res], axis=1), self.reference_point
            )
            ep_returns = ep_returns.at[:, t].set(self.gamma * ep_returns[:, t - 1] + r)
            ep_ret += r
            ep_len += 1

            # Log transition
            buffer.store(states, a, r, v, logp, res)  # Use un-normalized states

            # Update state
            states = self.env.get_states(self.reference_point)

            # Update extrinsics
            if self.env.use_adaptive_approach is True:
                actual_wrench = self.env.get_actual_wrench()
                desired_wrench = self.env.get_desired_wrench(a)
                res = actual_wrench - desired_wrench
            else:
                res = jnp.empty((self.env.num_envs, 0))

            # Check if a timeout is appropriate
            timeout = ep_len == self.max_ep_len
            epoch_ended = t == self.steps_per_epoch - 1

            # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
            # for each env individually
            if terminal.all() or timeout or epoch_ended:
                # If the trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = self.agent.act(
                        jnp.concatenate([states, res], axis=1)
                    )  # Use un-normalized states
                else:
                    v = jnp.zeros(self.env.num_envs)

                buffer.end_traj(v)

                self.env.reset()
                states, ep_ret, ep_len = (
                    self.env.get_states(self.reference_point),
                    jnp.zeros(self.env.num_envs),
                    0,
                )

                if self.env.use_adaptive_approach is True:
                    res = jnp.zeros((self.env.num_envs, self.env.res_dim))
                else:
                    res = jnp.empty((self.env.num_envs, 0))

        # Get the data from the training loop and save it
        data = buffer.get()
        save_training_data(self.ckpt_dir, self.pretraining_data_file_name, data)

    def _build_adaptation_module(self):
        if self.env.am_architecture == "transformer":
            return TransformerAdaptationModule(
                self.env.history_len, self.state_action_dim, self.env.ext_dim
            )
        if self.env.am_architecture == "cnn":
            return CNNAdaptationModule(
                self.env.history_len, self.state_action_dim, self.env.ext_dim
            )
        raise ValueError(
            f"Unknown adaptation module architecture '{self.env.am_architecture}'."
        )

    def _select_am_loss_fn(self):
        if self.env.am_architecture == "transformer":
            kl_weight = float(self.am_kl_weight)

            def _loss(model, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
                mu, log_sigma = jax.vmap(lambda hist: model(hist, return_stats=True))(X)
                target = y[:, -1, :]
                log_sigma = jnp.clip(log_sigma, -6.0, 2.0)
                sigma_sq = jnp.exp(2.0 * log_sigma)
                nll = 0.5 * jnp.sum(
                    ((target - mu) ** 2) / sigma_sq + 2.0 * log_sigma, axis=-1
                )
                if kl_weight > 0.0:
                    kl = 0.5 * jnp.sum(
                        mu**2 + sigma_sq - 1.0 - jnp.log(sigma_sq + 1e-8), axis=-1
                    )
                    return jnp.mean(nll + kl_weight * kl)
                return jnp.mean(nll)

            return _loss

        def _mse_loss(model, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            preds = jax.vmap(model)(X)
            target = y[:, -1, :]
            mse = jnp.square(preds - target)
            return jnp.mean(jnp.sum(mse, axis=-1))

        return _mse_loss

    def _build_sliding_windows(
        self,
        features: jnp.ndarray,
        targets: jnp.ndarray,
        mask: jnp.ndarray,
        seq_len: int,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Construct stride-1 windows of length ``seq_len`` and keep only entries with a fully populated history.
        """
        # Discard early timesteps where the history buffer is still warming up.
        valid_mask = mask.at[: seq_len - 1, :].set(False)
        valid_count = int(valid_mask.sum())
        if valid_count == 0:
            return (
                jnp.empty((0, seq_len, features.shape[2]), dtype=features.dtype),
                jnp.empty((0, seq_len, targets.shape[2]), dtype=targets.dtype),
            )

        # Flatten (time, env) indices where a full history is available.
        valid_t, valid_env = jnp.where(
            valid_mask,
            size=int(valid_mask.size),
            fill_value=-1,
        )
        valid_t = valid_t[:valid_count]
        valid_env = valid_env[:valid_count]
        starts = valid_t - (seq_len - 1)

        feat_dim = features.shape[2]
        target_dim = targets.shape[2]

        def _slice_single(start: jnp.ndarray, env_idx: jnp.ndarray):
            start = jnp.asarray(start, dtype=jnp.int32)
            env_idx = jnp.asarray(env_idx, dtype=jnp.int32)
            feat_slice = jax.lax.dynamic_slice(
                features,
                (start, env_idx, 0),
                (seq_len, 1, feat_dim),
            )
            feat_slice = jnp.squeeze(feat_slice, axis=1)
            target_slice = jax.lax.dynamic_slice(
                targets,
                (start, env_idx, 0),
                (seq_len, 1, target_dim),
            )
            target_slice = jnp.squeeze(target_slice, axis=1)
            return feat_slice, target_slice

        feat_windows, target_windows = jax.vmap(_slice_single)(starts, valid_env)
        return feat_windows, target_windows

    def _compute_mean_errors(
        self,
        obs: jnp.ndarray,
        actual_wrench: jnp.ndarray | None = None,
        ext: jnp.ndarray | None = None,
        mask: jnp.ndarray | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """
        Compute mean tracking, attitude, and extrinsic errors for the current step.
        """

        tracking = calc_lateral_tracking_error(obs, self.planner)
        attitude = jnp.degrees(calc_attitude_error(obs))
        if mask is not None:
            mask_f = jnp.asarray(mask, dtype=tracking.dtype)
            denom = jnp.maximum(mask_f.sum(), 1.0)
            tracking_mean = (tracking * mask_f).sum() / denom
            attitude_mean = (attitude * mask_f).sum() / denom
        else:
            tracking_mean = tracking.mean()
            attitude_mean = attitude.mean()

        extrinsic_mean = None
        if (
            self.env.use_adaptive_approach
            and actual_wrench is not None
            and ext is not None
        ):
            extrinsic_error = calc_extrinsic_error(ext, actual_wrench)
            if mask is not None:
                mask_f = jnp.asarray(mask, dtype=extrinsic_error.dtype)
                denom = jnp.maximum(mask_f.sum(), 1.0)
                extrinsic_mean = (extrinsic_error * mask_f).sum() / denom
            else:
                extrinsic_mean = extrinsic_error.mean()

        return tracking_mean, attitude_mean, extrinsic_mean

    def _load_rl_hyperparams(self) -> None:
        """
        Load the relevant hyperparams from the config file.
        """
        if isinstance(self.agent, VPG):
            self.agent._load_vpg_hyperparams()
        elif isinstance(self.agent, PPO):
            self.agent._load_ppo_hyperparams()
        else:
            raise Exception("Agent has not been implemented.")

        # Mirror agent hyperparameters locally for convenience
        self.steps_per_epoch = self.agent.steps_per_epoch
        self.epochs = self.agent.epochs
        self.max_ep_len = self.agent.max_ep_len
        self.gamma = self.agent.gamma
        self.lam = self.agent.lam

        self.episode_len = self.env.env_cfg.control.RL.episode_len
        self.n_evals = self.env.env_cfg.control.RL.n_evals

    def _create_checkpoint_file_names(self) -> None:
        """
        Create the checkpoint file names for the pretraining and training data and states.
        """

        # Build filename components
        adaptive = "adaptive" if self.env.use_adaptive_approach else None
        pretrained = "pretrained" if self.env.use_pretrained else None
        nominal = None if self.env.train_with_failures else "nominal"

        def build_name(prefix: str) -> str:
            parts = [prefix, adaptive, pretrained, nominal]
            return "_".join(p for p in parts if p) + ".pkl"

        # Pretraining filenames
        if adaptive:
            self.pretraining_data_file_name = "pretraining_data_adaptive.pkl"
            self.pretraining_state_file_name = "pretraining_state_adaptive.pkl"
        else:
            self.pretraining_data_file_name = "pretraining_data.pkl"
            self.pretraining_state_file_name = "pretraining_state.pkl"

        # Training filenames
        self.training_data_file_name = build_name("training_data")
        self.training_state_file_name = build_name("training_state")
        self.adaptation_module_file_name = (
            f"adapt_module_state_{self.env.am_architecture}.pkl"
        )
