import os
import warnings

import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb

from smallsat_sim.envs.vec_env import VecEnv
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
    load_training_data,
)
from smallsat_sim.controllers.rl.runners.evaluation_loop import evaluate_runner
from smallsat_sim.controllers.rl.runners.training_loop import learn_runner
from smallsat_sim.controllers.rl.runners.adaptation_training import (
    train_adaptation_module_on_policy_runner,
)
from smallsat_sim.controllers.rl.runners.adaptive_context import build_adaptive_context
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

        self.rl_cfg = self.env.env_cfg.control.RL

        # Vectorize adaptation module
        self.adaptation_module = jax.vmap(self.am, in_axes=(0, 0))
        self.am_kl_weight = self.env.env_cfg.control.RL.am_kl_weight
        self.am_loss_fn = self._select_am_loss_fn()
        self.am_loss_components_fn = self._select_am_loss_components_fn()

        # JIT-compile the adaptation module updates
        self.jitted_batched_am_loss_and_grad = nnx.jit(
            nnx.value_and_grad(self.am_loss_fn), static_argnums=()
        )
        self.jitted_batched_am_loss_components = nnx.jit(
            self.am_loss_components_fn, static_argnums=()
        )

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
        learn_runner(self)

    def train_adaptation_module_on_policy(self) -> None:
        """
        Train adaptation module to predict extrinsics from the history of states and actions with
        on-policy data (RMA approach).
        NOTE: use_adaptive_approach must be set to True in the environment config.
        """
        train_adaptation_module_on_policy_runner(self)

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
        evaluate_runner(self, phase=phase)

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
                res = build_adaptive_context(
                    commanded_ctrl=a,
                    applied_ctrl=self.env.mjx_batch.ctrl,
                    actual_wrench=actual_wrench,
                    desired_wrench=desired_wrench,
                    previous_context=res,
                    use_adaptive_approach=True,
                    adaptive_context_mode=self.env.adaptive_context_mode,
                    thruster_mixer_T=self.env._thruster_mixer_T,
                )
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
        rngs = nnx.Rngs(params=self._take_keys(), dropout=self._take_keys())
        predict_delta_dim = (
            self.env.obs_dim
            if float(self.env.env_cfg.control.RL.am_predict_delta_weight) > 0.0
            else 0
        )
        predict_tracking = (
            float(self.env.env_cfg.control.RL.am_predict_tracking_weight) > 0.0
        )
        if self.env.am_architecture == "transformer":
            return TransformerAdaptationModule(
                self.env.history_len,
                self.state_action_dim,
                self.env.ext_dim,
                query_dim=self.env.am_query_dim,
                predict_delta_dim=predict_delta_dim,
                predict_tracking=predict_tracking,
                rngs=rngs,
            )
        if self.env.am_architecture == "cnn":
            return CNNAdaptationModule(
                self.env.history_len,
                self.state_action_dim,
                self.env.ext_dim,
                query_dim=self.env.am_query_dim,
                predict_delta_dim=predict_delta_dim,
                predict_tracking=predict_tracking,
                rngs=rngs,
            )
        raise ValueError(
            f"Unknown adaptation module architecture '{self.env.am_architecture}'."
        )

    def _select_am_loss_components_fn(self):
        if self.env.am_architecture == "transformer":
            kl_weight = float(self.am_kl_weight)
            delta_weight = float(self.rl_cfg.am_predict_delta_weight)
            tracking_weight = float(self.rl_cfg.am_predict_tracking_weight)
            res_dim = int(self.env.res_dim)
            query_dim = int(self.env.am_query_dim)
            obs_dim = int(self.env.obs_dim)

            def _components(model, X: jnp.ndarray, y: jnp.ndarray):
                target_all = y[:, -1, :]
                target = target_all[:, :res_dim]
                query = target_all[:, res_dim : res_dim + query_dim]
                delta_target = target_all[
                    :, res_dim + query_dim : res_dim + query_dim + obs_dim
                ]
                tracking_target = target_all[
                    :, res_dim + query_dim + obs_dim : res_dim + query_dim + obs_dim + 1
                ]
                mu, log_sigma, delta_pred, tracking_pred = jax.vmap(
                    lambda hist, query_i: model(
                        hist,
                        query_i,
                        return_stats=True,
                        return_predictions=True,
                        training=True,
                    )
                )(X, query)
                log_sigma = jnp.clip(log_sigma, -6.0, 2.0)
                sigma_sq = jnp.exp(2.0 * log_sigma)
                context_loss = jnp.mean(
                    0.5
                    * jnp.sum(
                        ((target - mu) ** 2) / sigma_sq + 2.0 * log_sigma,
                        axis=-1,
                    )
                )
                delta_loss = jnp.array(0.0, dtype=context_loss.dtype)
                if delta_weight > 0.0:
                    delta_scale = jnp.mean(jnp.square(delta_target)) + 1e-6
                    delta_loss = (
                        jnp.mean(jnp.square(delta_pred - delta_target)) / delta_scale
                    )
                tracking_loss = jnp.array(0.0, dtype=context_loss.dtype)
                if tracking_weight > 0.0:
                    tracking_scale = jnp.mean(jnp.square(tracking_target)) + 1e-6
                    tracking_loss = jnp.mean(
                        jnp.square(tracking_pred - tracking_target)
                    ) / tracking_scale
                kl_loss = jnp.array(0.0, dtype=context_loss.dtype)
                if kl_weight > 0.0:
                    kl_loss = jnp.mean(
                        0.5
                        * jnp.sum(
                            mu**2 + sigma_sq - 1.0 - jnp.log(sigma_sq + 1e-8),
                            axis=-1,
                        )
                    )
                total = (
                    context_loss
                    + delta_weight * delta_loss
                    + tracking_weight * tracking_loss
                    + kl_weight * kl_loss
                )
                return context_loss, delta_loss, tracking_loss, kl_loss, total

            return _components

        delta_weight = float(self.rl_cfg.am_predict_delta_weight)
        tracking_weight = float(self.rl_cfg.am_predict_tracking_weight)
        res_dim = int(self.env.res_dim)
        query_dim = int(self.env.am_query_dim)
        obs_dim = int(self.env.obs_dim)

        def _mse_components(model, X: jnp.ndarray, y: jnp.ndarray):
            target_all = y[:, -1, :]
            target = target_all[:, :res_dim]
            query = target_all[:, res_dim : res_dim + query_dim]
            delta_target = target_all[
                :, res_dim + query_dim : res_dim + query_dim + obs_dim
            ]
            tracking_target = target_all[
                :, res_dim + query_dim + obs_dim : res_dim + query_dim + obs_dim + 1
            ]
            preds, delta_pred, tracking_pred = jax.vmap(
                lambda hist, query_i: model(
                    hist,
                    query_i,
                    return_predictions=True,
                )
            )(X, query)
            context_loss = jnp.mean(jnp.sum(jnp.square(preds - target), axis=-1))
            delta_loss = jnp.array(0.0, dtype=context_loss.dtype)
            if delta_weight > 0.0:
                delta_scale = jnp.mean(jnp.square(delta_target)) + 1e-6
                delta_loss = (
                    jnp.mean(jnp.square(delta_pred - delta_target)) / delta_scale
                )
            tracking_loss = jnp.array(0.0, dtype=context_loss.dtype)
            if tracking_weight > 0.0:
                tracking_scale = jnp.mean(jnp.square(tracking_target)) + 1e-6
                tracking_loss = (
                    jnp.mean(jnp.square(tracking_pred - tracking_target))
                    / tracking_scale
                )
            kl_loss = jnp.array(0.0, dtype=context_loss.dtype)
            total = (
                context_loss
                + delta_weight * delta_loss
                + tracking_weight * tracking_loss
            )
            return context_loss, delta_loss, tracking_loss, kl_loss, total

        return _mse_components

    def _select_am_loss_fn(self):
        components_fn = self._select_am_loss_components_fn()

        def _loss(model, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
            return components_fn(model, X, y)[-1]

        return _loss

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
            estimated_wrench = ext[:, :6]
            if estimated_wrench.shape == actual_wrench.shape:
                estimated_wrench = estimated_wrench + self.env.get_desired_wrench(
                    jnp.zeros((ext.shape[0], self.env.act_dim), dtype=ext.dtype)
                )
            extrinsic_error = calc_extrinsic_error(estimated_wrench, actual_wrench)
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
        context_mode = (
            self.env.adaptive_context_mode
            if self.env.use_adaptive_approach
            else None
        )
        pretrained = "pretrained" if self.env.use_pretrained else None
        nominal = None if self.env.train_with_failures else "nominal"

        def build_name(prefix: str) -> str:
            parts = [prefix, adaptive, context_mode, pretrained, nominal]
            predictive_am = (
                self.env.use_task_conditioned_am
                or float(self.env.env_cfg.control.RL.am_predict_delta_weight) > 0.0
                or float(self.env.env_cfg.control.RL.am_predict_tracking_weight) > 0.0
            )
            if prefix.startswith("adapt_module") and predictive_am:
                parts.append("taskpred")
            return "_".join(p for p in parts if p) + ".pkl"

        # Pretraining filenames
        if adaptive:
            self.pretraining_data_file_name = build_name("pretraining_data")
            self.pretraining_state_file_name = build_name("pretraining_state")
        else:
            self.pretraining_data_file_name = "pretraining_data.pkl"
            self.pretraining_state_file_name = "pretraining_state.pkl"

        # Training filenames
        self.training_data_file_name = build_name("training_data")
        self.training_state_file_name = build_name("training_state")
        self.adaptation_module_file_name = build_name(
            f"adapt_module_state_{self.env.am_architecture}"
        )
