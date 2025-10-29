import os
import time
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
from smallsat_sim.controllers.rl.modules.adaptation_module import AdaptationModule
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
    batch_mse_loss_fn,
    mae_loss_fn,
    normalize_obs,
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
        self.am = AdaptationModule(50, env.obs_dim + env.act_dim, env.ext_dim)
        self.reference_point = planner.get_reference(
            self.env.get_obs()
        )  # Is re-used in every episode
        self.pd_ctrl = VectorizedPDController(env, planner)
        self._load_rl_hyperparams()

        # Vectorize adaptation module
        self.adaptation_module = jax.vmap(self.am)

        # JIT-compile the adaptation module updates
        self.jitted_batched_am_loss_and_grad = nnx.jit(
            nnx.value_and_grad(batch_mse_loss_fn), static_argnums=()
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
                name=self.env.run_name,
                config={
                    "num_envs": self.env.num_envs,
                    "steps_per_epoch": self.steps_per_epoch,
                    "epochs": self.epochs,
                    "max_ep_len": self.max_ep_len,
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "actor_lr": self.actor_lr,
                    "critic_lr": self.critic_lr,
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
            extrinsics = pretraining_data["extrinsics"].reshape(-1, self.env.ext_dim)
        else:
            extrinsics = jnp.empty((self.steps_per_epoch * self.env.num_envs, 0))

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
                jnp.concatenate([obs, extrinsics], axis=1),
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
            self.agent.update_policy_gradient(
                self._take_keys(), obs, act_clipped, tdres, logp
            )
        else:
            raise Exception(
                "This strategy does not exist. Options are [supervised_learning] and [rl]."
            )

        # Pretrain the base network
        self.agent.update_value_function(
            self._take_keys(),
            jnp.concatenate([obs, extrinsics], axis=1),
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
                nnx.update(
                    self.agent.actor.mu_net, restored_state["actor_model"].mu_net
                )
                nnx.update(
                    self.agent.critic.v_net, restored_state["critic_model"].v_net
                )
            else:
                print("No pretrained modules available.\n")

        print("Training agent...\n")

        # Set up buffer
        buffer = ReplayBuffer(
            self.env.num_envs,
            self.env.obs_dim,
            self.env.act_dim,
            self.env.ext_dim,
            self.steps_per_epoch,
            self.gamma,
            self.lam,
        )

        # Initialize the environment
        self.env.reset()
        self.env.reset_perturbations()
        states, ep_ret, ep_len = (
            self.env.get_states(self.reference_point),
            jnp.zeros(self.env.num_envs),
            0,
        )
        states_normalized = states
        episode_counter = 0

        if self.env.use_adaptive_approach is True:
            ext = jnp.ones_like(self.env.mjx_batch.ctrl)  # No control input yet
        else:
            ext = jnp.empty((self.env.num_envs, 0))

        # Create PRNG keys
        subkeys_train = self._take_keys(self.epochs)

        # Main training loop
        for epoch in range(self.epochs):
            # Apply perturbations ramp-up
            if self.env.train_with_failures and epoch >= self.epochs // 2:
                ramp_duration = max(self.epochs // 2, 1)
                ramp_progress = min((epoch - ramp_duration) / ramp_duration, 1.0)
                self.env.reset_perturbations()  # avoid accumulating failures across epochs
                self.env.apply_random_perturbations(
                    key=subkeys_train[epoch],
                    fraction_perturbed_envs=0.05
                    + (0.5 - 0.05) * max(ramp_progress, 0.0),
                )
                self.env.apply_random_disturbance(
                    key=subkeys_train[epoch],
                    fraction_disturbed_envs=0.05
                    + (0.15 - 0.05) * max(ramp_progress, 0.0),
                )

            # Epoch variables
            ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
            ep_obs = jnp.zeros(
                (self.env.num_envs, self.steps_per_epoch, self.env.obs_dim)
            )

            # For logging mean errors
            ep_mean_tracking_error = jnp.zeros(
                (self.env.num_envs, self.steps_per_epoch)
            )
            ep_mean_angle_error = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
            ep_mean_extrinsic_error = jnp.zeros(
                (self.env.num_envs, self.steps_per_epoch)
            )

            for t in range(self.steps_per_epoch):
                # Get actions from the agent
                a, v, logp = self.agent.act(
                    jnp.concatenate([states, ext], axis=1), log=True
                )  # Use un-normalized states

                # Compute and log mean errors
                obs = self.env.get_obs()
                tracking_mean, angle_mean, extrinsic_mean = self._compute_mean_errors(
                    obs, a, ext
                )
                ep_mean_tracking_error = ep_mean_tracking_error.at[:, t].set(
                    tracking_mean
                )
                ep_mean_angle_error = ep_mean_angle_error.at[:, t].set(angle_mean)
                if (
                    self.env.use_adaptive_approach is True
                    and extrinsic_mean is not None
                ):
                    ep_mean_extrinsic_error = ep_mean_extrinsic_error.at[:, t].set(
                        extrinsic_mean
                    )

                # Perform environment transition
                r, terminal = self.env.transition(
                    a, states, self.reference_point, epoch
                )
                ep_returns = ep_returns.at[:, t].set(
                    self.gamma * ep_returns[:, t - 1] + r
                )
                ep_ret += r
                ep_len += 1

                # Log transition
                buffer.store(states, a, r, v, logp, ext)  # Use un-normalized states

                # Update state
                states = self.env.get_states(self.reference_point)
                ep_obs = ep_obs.at[:, t, :].set(states)
                states_normalized = normalize_obs(states, ep_obs, t)

                # Update extrinsics
                if self.env.use_adaptive_approach is True:
                    ext = self.env.mjx_batch.ctrl / (
                        a + 1e-8 * jnp.ones_like(self.env.mjx_batch.ctrl)
                    )
                else:
                    ext = jnp.empty((self.env.num_envs, 0))

                # Check if a timeout is appropriate
                timeout = ep_len == self.max_ep_len
                epoch_ended = t == self.steps_per_epoch - 1

                # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
                # for each env individually
                if terminal.all() or timeout or epoch_ended:
                    # If the trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.agent.act(
                            jnp.concatenate([states, ext], axis=1)
                        )  # Use un-normalized states
                    else:
                        v = jnp.zeros(self.env.num_envs)

                    buffer.end_traj(v)

                    # Log the mean scaled episodic returns
                    if self.env.use_wandb:
                        wandb.log(
                            {
                                "mean_episodic_returns": ep_ret.mean(),
                            }
                        )
                    if self.agent.has_logger:
                        self.env.logger.log(
                            self.env.run_id,
                            float(self.env.mjx_batch.time[0]),
                            step=episode_counter,
                            run_name=self.env.run_name,
                            stage="policy_training",
                            mean_episodic_returns=ep_ret.mean(),
                        )

                    self.env.reset()
                    self.env.reset_perturbations()
                    states, ep_ret, ep_len = (
                        self.env.get_states(self.reference_point),
                        jnp.zeros(self.env.num_envs),
                        0,
                    )
                    states_normalized = states

                    if self.env.use_adaptive_approach is True:
                        ext = jnp.ones_like(
                            self.env.mjx_batch.ctrl
                        )  # No control input yet
                    else:
                        ext = jnp.empty((self.env.num_envs, 0))

                    episode_counter += 1

            # Get the data from the training loop and save it
            data = buffer.get()
            save_training_data(self.ckpt_dir, self.training_data_file_name, data)

            obs = data["obs"].reshape(-1, self.env.obs_dim)
            actions = data["act"].reshape(-1, self.env.act_dim)
            rews = data["rews"].reshape(-1)
            tdres = data["tdres"].reshape(-1)
            returns = data["ret"].reshape(-1)
            logp = data["logp"].reshape(-1)
            if self.env.use_adaptive_approach is True:
                extrinsics = data["extrinsics"].reshape(-1, self.env.ext_dim)
            else:
                extrinsics = jnp.empty((self.steps_per_epoch * self.env.num_envs, 0))

            # # Policy gradient update
            # actor_loss = self.agent.update_policy_gradient(
            #     subkeys_train[epoch],
            #     jnp.concatenate([obs, extrinsics], axis=1),
            #     actions,
            #     tdres,
            #     logp,
            # )

            # # Value function updates
            # critic_loss = self.agent.update_value_function(
            #     subkeys_train[epoch],
            #     jnp.concatenate([obs, extrinsics], axis=1),
            #     returns,
            # )

            # Update the policy gradient and the value function
            actor_loss, critic_loss = self.agent.update_actor_critic_minibatch(
                subkeys_train[epoch],
                jnp.concatenate([obs, extrinsics], axis=1),
                actions,
                tdres,
                logp,
                returns,
            )

            # Monitor key RL metrics during training using Weights & Biases
            if self.env.use_wandb:
                wandb.log(
                    {
                        "mean_rewards": rews.mean(),
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "num_terminal": jnp.sum(terminal),
                        "mean_log_std": self.agent.actor.log_std.value.mean(),
                        "mean_std": jnp.exp(self.agent.actor.log_std.value).mean(),
                    }
                )

            # Log key RL metrics
            if self.agent.has_logger:
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    step=int(epoch),
                    run_name=self.env.run_name,
                    stage="policy_training",
                    mean_rewards=rews.mean(),
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    num_terminal=jnp.sum(terminal),
                    mean_log_std=self.agent.actor.log_std.value.mean(),
                    mean_std=jnp.exp(self.agent.actor.log_std.value).mean(),
                    mean_tracking_error=ep_mean_tracking_error.mean(),
                    mean_angle_error=ep_mean_angle_error.mean(),
                    mean_extrinsic_error=ep_mean_extrinsic_error.mean(),
                )

            # Save the trained actor and critic network weights
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

        # Check if adaptation module has already been trained
        file_path = os.path.join(self.ckpt_dir, "adapt_module_state.pkl")
        if os.path.isfile(file_path):
            return

        # Check if trained actor and critic modules are available and load them
        file_path = os.path.join(self.ckpt_dir, self.training_state_file_name)
        if os.path.isfile(file_path):
            restored_state = load_trained_modules(
                self.ckpt_dir, self.training_state_file_name
            )
            nnx.update(self.agent.actor.mu_net, restored_state["actor_model"].mu_net)
            nnx.update(self.agent.critic.v_net, restored_state["critic_model"].v_net)
        else:
            raise Exception(
                "The base policy must be trained before the adaptation module."
            )

        print("Training adaptation module...\n")

        # Set up buffer
        buffer = ReplayBuffer(
            self.env.num_envs,
            self.env.obs_dim,
            self.env.act_dim,
            self.env.ext_dim,
            self.steps_per_epoch,
            self.gamma,
            self.lam,
        )

        # Initialize the environment
        self.env.reset()
        self.env.reset_perturbations()
        states, ep_ret, ep_len = (
            self.env.get_states(self.reference_point),
            jnp.zeros(self.env.num_envs),
            0,
        )
        states_normalized = states

        state_action_history = jnp.zeros(
            (self.env.num_envs, 50, self.env.obs_dim + self.env.act_dim)
        )  # No history in the beginning
        ext = jnp.ones_like(self.env.mjx_batch.ctrl)  # No control input yet
        ext_gt = jnp.ones_like(self.env.mjx_batch.ctrl)  # Ground truth extrinsics

        # Create PRNG keys
        subkeys_train = self._take_keys(self.epochs)

        # Main training loop
        for epoch in range(self.epochs):
            ramp_duration = max(self.epochs // 2, 1)
            ramp_progress = min((epoch - ramp_duration) / ramp_duration, 1.0)
            self.env.reset_perturbations()  # avoid accumulating failures across epochs
            self.env.apply_random_perturbations(
                key=subkeys_train[epoch],
                fraction_perturbed_envs=0.05 + (0.5 - 0.05) * max(ramp_progress, 0.0),
            )
            self.env.apply_random_disturbance(
                key=subkeys_train[epoch],
                fraction_disturbed_envs=0.05 + (0.15 - 0.05) * max(ramp_progress, 0.0),
            )

            ep_obs = jnp.zeros(
                (self.env.num_envs, self.steps_per_epoch, self.env.obs_dim)
            )
            ep_mean_tracking_error = jnp.zeros(
                (self.env.num_envs, self.steps_per_epoch)
            )
            ep_mean_angle_error = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
            ep_mean_extrinsic_error = jnp.zeros(
                (self.env.num_envs, self.steps_per_epoch)
            )
            for t in range(self.steps_per_epoch):
                # Get actions from the agent
                a = self.agent.get_control_input(
                    self.env, "am_training", jnp.concatenate([states, ext], axis=1)
                )  # Use un-normalized states

                # Update state-action history
                state_action_history = jnp.roll(state_action_history, shift=-1, axis=1)
                state_action_history = state_action_history.at[:, -1, :].set(
                    jnp.concatenate([states, a], axis=1)
                )

                # Compute and log mean errors
                obs = self.env.get_obs()
                tracking_mean, angle_mean, extrinsic_mean = self._compute_mean_errors(
                    obs, a, ext
                )
                ep_mean_tracking_error = ep_mean_tracking_error.at[:, t].set(
                    tracking_mean
                )
                ep_mean_angle_error = ep_mean_angle_error.at[:, t].set(angle_mean)
                if extrinsic_mean is not None:
                    ep_mean_extrinsic_error = ep_mean_extrinsic_error.at[:, t].set(
                        extrinsic_mean
                    )

                # Perform environment transition
                _, terminal = self.env.transition(
                    a, states, self.reference_point, epoch
                )
                ep_len += 1

                # Log transition
                placeholder = jnp.zeros(self.env.num_envs)
                buffer.store(
                    states, a, placeholder, placeholder, placeholder, ext_gt
                )  # Use un-normalized states

                # Update state
                states = self.env.get_states(self.reference_point)
                ep_obs = ep_obs.at[:, t, :].set(states)
                states_normalized = normalize_obs(states, ep_obs, t)

                # Update extrinsics
                ext_gt = self.env.mjx_batch.ctrl / (
                    a + 1e-8 * jnp.ones_like(self.env.mjx_batch.ctrl)
                )
                ext = self.adaptation_module(state_action_history)

                # Check if a timeout is appropriate
                timeout = ep_len == self.max_ep_len
                epoch_ended = t == self.steps_per_epoch - 1

                # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
                # for each env individually
                if terminal.all() or timeout or epoch_ended:
                    self.env.reset()
                    self.env.reset_perturbations()
                    states, ep_ret, ep_len = (
                        self.env.get_states(self.reference_point),
                        jnp.zeros(self.env.num_envs),
                        0,
                    )
                    states_normalized = states

                    ext = jnp.ones_like(self.env.mjx_batch.ctrl)  # No control input yet
                    ext_gt = jnp.ones_like(self.env.mjx_batch.ctrl)

            # Get the data from the training loop
            data = buffer.get()

            obs = data["obs"]
            act = data["act"]
            extrinsics = data["extrinsics"]

            state_action_data = jnp.concatenate([obs, act], axis=2)

            num_nn_epochs = 100

            # Optimizer
            am_lr = self.env.env_cfg.control.RL.am_lr
            self.am_optimizer = nnx.Optimizer(
                self.am,
                optax.adam(
                    learning_rate=am_lr,
                    eps=1e-5,
                ),
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
                state_action_data,
                extrinsics,
                key=split_key,
                shuffle=False,
                trim_for_cnn=True,
            )

            # Training loop
            for nn_epoch in range(num_nn_epochs):
                # Compute the loss
                am_train_loss, grads = self.jitted_batched_am_loss_and_grad(
                    self.am,
                    X_train,
                    y_train,
                )
                print(f"{am_train_loss = }\n")
                self.am_optimizer.update(grads)

                # Periodically evaluate on the validation set (e.g., every 10 nn_epoch)
                if nn_epoch % 10 == 0:
                    am_val_loss, _ = self.jitted_batched_am_loss_and_grad(
                        self.am, X_val, y_val
                    )
                    print(
                        f"Epoch {nn_epoch}: Train Loss = {am_train_loss:.4f}, Val Loss = {am_val_loss:.4f}\n"
                    )

            # Monitor key RL metrics during training using Weights & Biases
            if self.env.use_wandb:
                wandb.log(
                    {
                        "am_train_loss": am_train_loss,
                        "am_val_loss": am_val_loss,
                    }
                )

            # Log key RL metrics
            if self.agent.has_logger:
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    step=int(epoch),
                    run_name=self.env.run_name,
                    stage="am_training",
                    am_train_loss=am_train_loss,
                    am_val_loss=am_val_loss,
                    mean_tracking_error=ep_mean_tracking_error.mean(),
                    mean_angle_error=ep_mean_angle_error.mean(),
                    mean_extrinsic_error=ep_mean_extrinsic_error.mean(),
                )

            # Save the adaptation module weights
            save_adaptation_module(self.am, self.ckpt_dir, "adapt_module_state.pkl")

    def posttrain(self) -> None:
        """
        Fine-tune the policy on imperfectly estimated extrinsics (phase 3 in A-RMA).
        NOTE: use_adaptive_approach must be set to True in the environment config.
        """
        raise NotImplementedError("Post-training is not implemented yet.\n")

    def evaluate(self, phase: int = 2) -> None:
        """
        Evaluate the agent.
        If phase == 1, evaluate base policy before training the adaptation module.
        If phase == 2, evaluate base policy after training the adaptation module.
        """
        print("Evaluating agent...\n")

        # Check if trained actor, critic and adaptation modules are available and load them
        file_path = os.path.join(self.ckpt_dir, self.training_state_file_name)
        if os.path.isfile(file_path):
            restored_state = load_trained_modules(
                self.ckpt_dir, self.training_state_file_name
            )
            nnx.update(self.agent.actor.mu_net, restored_state["actor_model"].mu_net)
            nnx.update(self.agent.critic.v_net, restored_state["critic_model"].v_net)
            if self.env.use_adaptive_approach and phase == 2:
                adapt_module_state = load_trained_modules(
                    self.ckpt_dir, "adapt_module_state.pkl"
                )
                nnx.update(self.am, adapt_module_state["am_model"])
        else:
            raise Exception("Not all necessary modules have been trained yet.\n")

        # PRNG keys for each eval
        subkeys_eval = self._take_keys(self.n_evals)
        subkeys_eval = jnp.atleast_2d(subkeys_eval)

        returns = jnp.zeros((self.env.num_envs, self.n_evals))

        for eval in range(self.n_evals):
            print(f"Testing policy: episode {eval+1}/{self.n_evals}\n")
            self.env.reset()
            self.env.reset_perturbations()
            states = self.env.get_states(self.reference_point)
            states_normalized = states

            state_action_history = jnp.zeros(
                (self.env.num_envs, 50, self.env.obs_dim + self.env.act_dim)
            )  # No history in the beginning
            if self.env.use_adaptive_approach is True:
                ext = jnp.ones_like(self.env.mjx_batch.ctrl)  # No control input yet
            else:
                ext = jnp.empty((self.env.num_envs, 0))

            # Initialize episode variables
            ep_ret = jnp.zeros(self.env.num_envs)
            ep_returns = jnp.zeros((self.env.num_envs, self.episode_len))
            ep_obs = jnp.zeros((self.env.num_envs, self.episode_len, self.env.obs_dim))
            terminal = jnp.zeros(self.env.num_envs, dtype=bool)
            ep_mean_tracking_error = jnp.zeros((self.env.num_envs, self.episode_len))
            ep_mean_angle_error = jnp.zeros((self.env.num_envs, self.episode_len))
            ep_mean_extrinsic_error = jnp.zeros((self.env.num_envs, self.episode_len))

            # Start perturbations halfway through the evaluation
            if self.env.train_with_failures and eval >= self.n_evals // 2:
                self.env.apply_random_perturbations(
                    key=subkeys_eval[eval],
                    fraction_perturbed_envs=0.5,
                )
                self.env.apply_random_disturbance(
                    key=subkeys_eval[eval],
                    fraction_disturbed_envs=0.15,
                )

            for ep in range(self.episode_len):
                # Get actions from the agent
                actions = self.agent.get_control_input(
                    self.env, "evaluation", jnp.concatenate([states, ext], axis=1)
                )  # Use un-normalized states

                # Update state-action history
                state_action_history = jnp.roll(state_action_history, shift=-1, axis=1)
                state_action_history = state_action_history.at[:, -1, :].set(
                    jnp.concatenate([states, actions], axis=1)
                )

                # Compute and log mean errors
                obs = self.env.get_obs()
                tracking_mean, angle_mean, extrinsic_mean = self._compute_mean_errors(
                    obs, actions, ext
                )
                ep_mean_tracking_error = ep_mean_tracking_error.at[:, ep].set(
                    tracking_mean
                )
                ep_mean_angle_error = ep_mean_angle_error.at[:, ep].set(angle_mean)
                if (
                    self.env.use_adaptive_approach is True
                    and extrinsic_mean is not None
                ):
                    ep_mean_extrinsic_error = ep_mean_extrinsic_error.at[:, ep].set(
                        extrinsic_mean
                    )

                # Update state
                states = self.env.get_states(self.reference_point)
                ep_obs = ep_obs.at[:, ep, :].set(states)
                states_normalized = normalize_obs(states, ep_obs, ep)

                # Update extrinsics
                if self.env.use_adaptive_approach is True:
                    if phase == 1:
                        ext = self.env.mjx_batch.ctrl / (
                            actions + 1e-8 * jnp.ones_like(self.env.mjx_batch.ctrl)
                        )
                    elif phase == 2:
                        ext = self.adaptation_module(state_action_history)
                    else:
                        raise Exception("There only exist two training phases.")
                else:
                    ext = jnp.empty((self.env.num_envs, 0))

                rewards, terminal = self.env.transition(
                    actions, states, self.reference_point
                )
                ep_returns = ep_returns.at[:, ep].set(
                    self.gamma * ep_returns[:, ep - 1] + rewards
                )
                ep_ret += rewards
                if terminal.all():  # Abort if all environments terminated
                    break

            returns = returns.at[:, eval].set(ep_ret)

            # Log key RL metrics
            if self.agent.has_logger:
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    step=int(eval),
                    run_name=self.env.run_name,
                    stage="evaluation",
                    mean_episodic_returns=ep_ret.mean(),
                    num_terminal=jnp.sum(terminal),
                    mean_tracking_error=ep_mean_tracking_error.mean(),
                    mean_angle_error=ep_mean_angle_error.mean(),
                    mean_extrinsic_error=ep_mean_extrinsic_error.mean(),
                )

        print(
            f"Average episodic return over all evals and all envs: {jnp.mean(returns)}\n"
        )

    def _generate_experience(self) -> None:
        """
        Roll out an episode where the actions are computed from a PD controller that serves as training data.
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
            self.env.ext_dim,
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
        states_normalized = states

        if self.env.use_adaptive_approach is True:
            ext = jnp.ones_like(self.env.mjx_batch.ctrl)  # No control input yet
        else:
            ext = jnp.empty((self.env.num_envs, 0))

        # Epoch variables
        ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
        ep_obs = jnp.zeros((self.env.num_envs, self.steps_per_epoch, self.env.obs_dim))

        # Main training loop
        for t in range(self.steps_per_epoch):
            # Get value estimates from the agent
            _, v, logp = self.agent.act(
                jnp.concatenate([states, ext], axis=1)
            )  # Use un-normalized states
            self.env.obs = self.env.get_obs()

            # Compute actions from the PD controller
            a = self.pd_ctrl.get_control_input(self.env)

            # Perform environment transition
            r, terminal = self.env.transition(a, states, self.reference_point)
            ep_returns = ep_returns.at[:, t].set(self.gamma * ep_returns[:, t - 1] + r)
            ep_ret += r
            ep_len += 1

            # Log transition
            buffer.store(states, a, r, v, logp, ext)  # Use un-normalized states

            # Update state
            states = self.env.get_states(self.reference_point)
            ep_obs = ep_obs.at[:, t, :].set(states)
            states_normalized = normalize_obs(states, ep_obs, t)

            # Update extrinsics
            if self.env.use_adaptive_approach is True:
                ext = self.env.mjx_batch.ctrl / (
                    a + 1e-8 * jnp.ones_like(self.env.mjx_batch.ctrl)
                )
            else:
                ext = jnp.empty((self.env.num_envs, 0))

            # Check if a timeout is appropriate
            timeout = ep_len == self.max_ep_len
            epoch_ended = t == self.steps_per_epoch - 1

            # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
            # for each env individually
            if terminal.all() or timeout or epoch_ended:
                # If the trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = self.agent.act(
                        jnp.concatenate([states, ext], axis=1)
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
                states_normalized = states

                if self.env.use_adaptive_approach is True:
                    ext = jnp.ones_like(self.env.mjx_batch.ctrl)  # No control input yet
                else:
                    ext = jnp.empty((self.env.num_envs, 0))

        # Get the data from the training loop and save it
        data = buffer.get()
        save_training_data(self.ckpt_dir, self.pretraining_data_file_name, data)

    def _compute_mean_errors(
        self,
        obs: jnp.ndarray,
        ctrl_input: jnp.ndarray,
        ext: jnp.ndarray,
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray | None]:
        """
        Compute mean tracking, attitude, and extrinsic errors for the current step.
        """

        tracking = calc_lateral_tracking_error(obs, self.planner)
        attitude = jnp.degrees(calc_attitude_error(obs))

        tracking_mean = tracking.mean()
        attitude_mean = attitude.mean()

        extrinsic_mean = None
        if self.env.use_adaptive_approach:
            ctrl_ratio = self.env.mjx_batch.ctrl / (
                ctrl_input + 1e-8 * jnp.ones_like(self.env.mjx_batch.ctrl)
            )
            extrinsic_error = calc_extrinsic_error(ext, ctrl_ratio)
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
