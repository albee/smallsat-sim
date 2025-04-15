import os
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.planners.base_planner import BasePlanner
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
    mse_loss_fn,
    mae_loss_fn,
    normalize_obs,
    scale_rews,
)
from smallsat_sim.utils.wandb_config import setup_wandb


class OnPolicyRunner(object):
    """
    On-policy runner for training and evaluation. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """

    def __init__(self, env: VecEnv, planner: BasePlanner) -> None:
        # Initialize the environment and agent
        self.env = env
        self.agent = PPO(self.env, planner)
        self.am = AdaptationModule(50, env.obs_dim + env.act_dim, env.ext_dim)
        self.reference_point = planner.reference_points[0]
        self.pd_ctrl = VectorizedPDController(env, planner)
        self._load_rl_hyperparams()

        # Path to save the checkpoints
        self.ckpt_dir = "smallsat_sim/controllers/rl/checkpoints/"

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

    def pretrain(self, strategy: str = "supervised_learning") -> None:
        """
        Pretrain the actor and critic networks.
        """
        file_path = os.path.join(self.ckpt_dir, self.pretraining_data_file_name)
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

        # Pretrain the policy network
        if strategy == "supervised_learning":
            # Clip the actions to the highest upper bound on the force range of the thrusters
            act_clipped = jnp.where(act > 0.6, 0.6, act)

            # Optimizer to pretrain the policy network
            actor_optimizer = nnx.Optimizer(
                self.agent.actor, optax.adam(learning_rate=1e-2, eps=1e-5)
            )

            # Split into training and validation sets
            X_train, y_train, X_val, y_val = train_val_split(
                jnp.concatenate([obs, extrinsics], axis=1), act_clipped
            )

            actor_losses = []
            actor_val_losses = []
            num_epochs = 80
            num_batches = 256
            batch_size = int(jnp.ceil(obs.shape[0] / num_batches))
            num_train_samples = X_train.shape[0]
            num_val_samples = X_val.shape[0]

            # Create PRNG keys
            key = jax.random.PRNGKey(42)
            keys = jax.random.split(key, num=num_epochs)

            # Training loop
            for epoch in range(num_epochs):
                # Shuffle the training data
                indices = jax.random.permutation(keys[epoch], num_train_samples)
                X_train = X_train[indices]
                y_train = y_train[indices]

                for i in range(0, num_train_samples, batch_size):
                    batch_X = X_train[i : i + batch_size]
                    batch_y = y_train[i : i + batch_size]

                    # Train network
                    actor_loss, grads = nnx.value_and_grad(mae_loss_fn)(
                        self.agent.actor, batch_X, batch_y, keys[epoch]
                    )
                    actor_losses.append(actor_loss)
                    actor_optimizer.update(grads)
                print(
                    f"Epoch: {epoch+1:2} avg. actor training loss: {sum(actor_losses)/len(actor_losses)}\n"
                )

                for i in range(0, num_val_samples, batch_size):
                    batch_X_val = X_val[i : i + batch_size]
                    batch_y_val = y_val[i : i + batch_size]

                    # Validate
                    actor_val_loss, _ = nnx.value_and_grad(mae_loss_fn)(
                        self.agent.actor, batch_X_val, batch_y_val, keys[epoch]
                    )
                    actor_val_losses.append(actor_val_loss)
                print(
                    f"Epoch: {epoch+1:2} avg. actor evaluation loss: {sum(actor_val_losses)/len(actor_val_losses)}\n"
                )
                if self.env.use_wandb:
                    wandb.log(
                        {
                            "training_loss": sum(actor_losses) / len(actor_losses),
                            "validation_loss": sum(actor_val_losses)
                            / len(actor_val_losses),
                        }
                    )
        elif strategy == "rl":
            self.agent.update_policy_gradient(
                jax.random.PRNGKey(42), obs, act_clipped, tdres, logp
            )
        else:
            raise Exception(
                "This strategy does not exist. Options are [supervised_learning] and [rl]."
            )

        # Pretrain the base network
        self.agent.update_value_function(
            jax.random.PRNGKey(42),
            jnp.concatenate([obs, extrinsics], axis=1),
            ret,
            minibatch=False,
        )

        # Save the trained actor and critic network weights
        save_trained_modules(
            self.agent, self.ckpt_dir, self.pretraining_state_file_name
        )

    def learn(self) -> None:
        """
        Main training loop.
        """
        print("Training agent...\n")

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

        # Create PRNG keys
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, num=self.epochs)

        # Main training loop
        for epoch in range(self.epochs):
            ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
            ep_obs = jnp.zeros(
                (self.env.num_envs, self.steps_per_epoch, self.env.obs_dim)
            )
            for t in range(self.steps_per_epoch):
                a, v, logp = self.agent.act(
                    jnp.concatenate([states, ext], axis=1), log=True
                )  # Use un-normalized states

                r, terminal = self.env.transition(a, states, epoch)
                ep_returns = ep_returns.at[:, t].set(
                    self.gamma * ep_returns[:, t - 1] + r
                )
                r_scaled = scale_rews(r, ep_returns, t)
                ep_ret += r_scaled
                ep_len += 1

                # Log transition
                buffer.store(
                    states, a, r_scaled, v, logp, ext
                )  # Use un-normalized states

                # Update state
                states = self.env.get_states(self.reference_point)
                ep_obs = ep_obs.at[:, t, :].set(states)
                states_normalized = normalize_obs(states, ep_obs, t)

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
                                "mean_scaled_episodic_returns": ep_ret.mean(),
                            }
                        )
                    if self.agent.has_logger:
                        self.env.logger.log(
                            self.env.run_id,
                            float(self.env.mjx_batch.time[0]),
                            run_name=self.env.run_name,
                            stage="policy_training",
                            mean_scaled_episodic_returns=ep_ret.mean(),
                        )

                    self.env.reset()
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
            #     keys[epoch],
            #     jnp.concatenate([obs, extrinsics], axis=1),
            #     actions,
            #     tdres,
            #     logp,
            # )

            # # Value function updates
            # critic_loss = self.agent.update_value_function(
            #     keys[epoch],
            #     jnp.concatenate([obs, extrinsics], axis=1),
            #     returns,
            # )

            # Update the policy gradient and the value function
            actor_loss, critic_loss = self.agent.update_actor_critic_minibatch(
                keys[epoch],
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
                        "mean_scaled_rewards": rews.mean(),
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "num_terminal": jnp.sum(terminal),
                        "mean_log_std": self.agent.actor.log_std.value.mean(),
                    }
                )

            # Log key RL metrics
            if self.agent.has_logger:
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    run_name=self.env.run_name,
                    stage="policy_training",
                    mean_scaled_rewards=rews.mean(),
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    num_terminal=jnp.sum(terminal),
                    mean_log_std=self.agent.actor.log_std.value.mean(),
                )

            # Save the trained actor and critic network weights
            save_trained_modules(
                self.agent, self.ckpt_dir, self.training_state_file_name
            )

    def train_adaptation_module_pretraining_data(self) -> None:
        """
        Train adaptation module to predict extrinsics from the history of states and actions from pretraining data.
        NOTE: use_adaptive_approach must be set to True in the environment config.
        """
        if self.env.use_adaptive_approach is False:
            return

        file_path = os.path.join(self.ckpt_dir, "adapt_module_state.pkl")
        if os.path.isfile(file_path):
            return

        print("Training adaptation module...\n")

        # Generate experience if necessary and load the pretraining data
        self._generate_experience()
        pretraining_data = load_training_data(
            self.ckpt_dir, self.pretraining_data_file_name
        )  # Can use pretraining data

        # Load the data
        obs = pretraining_data["obs"].reshape(-1, self.env.obs_dim)
        act = pretraining_data["act"].reshape(-1, self.env.act_dim)
        if self.env.use_adaptive_approach is True:
            extrinsics = pretraining_data["extrinsics"].reshape(-1, self.env.ext_dim)
        else:
            extrinsics = jnp.empty((self.steps_per_epoch * self.env.num_envs, 0))

        state_action_data = jnp.concatenate([obs, act], axis=1)

        key = jax.random.PRNGKey(42)
        num_epochs = 100

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
        X_train, y_train, X_val, y_val = train_val_split(state_action_data, extrinsics)

        # Training loop
        for epoch in range(num_epochs):
            # Compute the loss
            train_loss, grads = nnx.value_and_grad(mse_loss_fn)(
                self.am,
                X_train,
                y_train,
                key,
            )
            print(f"{train_loss = }\n")
            self.am_optimizer.update(grads)

            # Periodically evaluate on the validation set (e.g., every 10 epochs)
            if epoch % 10 == 0:
                val_loss = mse_loss_fn(self.am, X_val, y_val, key)
                print(
                    f"Epoch {epoch}: Train Loss = {train_loss:.4f}, Val Loss = {val_loss:.4f}\n"
                )

        # Save the adaptation module weights
        save_adaptation_module(self.am, self.ckpt_dir, "adapt_module_state.pkl")

    def train_adaptation_module_on_policy(self) -> None:
        """
        Train adaptation module to predict extrinsics from the history of states and actions with
        on-policy data (RMA approach).
        NOTE: use_adaptive_approach must be set to True in the environment config.
        """
        if self.env.use_adaptive_approach is False:
            return

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

        state_action_history = jnp.zeros(
            (self.env.num_envs, 50, self.env.obs_dim + self.env.act_dim)
        )  # No history in the beginning
        if self.env.use_adaptive_approach is True:
            ext = jnp.ones_like(self.env.mjx_batch.ctrl)  # No control input yet
        else:
            ext = jnp.empty((self.env.num_envs, 0))

        # Create PRNG keys
        key = jax.random.PRNGKey(42)

        # Main training loop
        for epoch in range(self.epochs):
            # ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
            ep_obs = jnp.zeros(
                (self.env.num_envs, self.steps_per_epoch, self.env.obs_dim)
            )
            for t in range(self.steps_per_epoch):
                a = self.agent.get_control_input(
                    "am_training", jnp.concatenate([states, ext], axis=1)
                )  # Use un-normalized states

                _, terminal = self.env.transition(a, states, epoch)
                ep_len += 1

                # Log transition
                placeholder = jnp.zeros((self.steps_per_epoch, self.env.num_envs))
                buffer.store(
                    states, a, placeholder, placeholder, placeholder, ext
                )  # Use un-normalized states

                # Update state
                states = self.env.get_states(self.reference_point)
                ep_obs = ep_obs.at[:, t, :].set(states)
                states_normalized = normalize_obs(states, ep_obs, t)

                state_action = jnp.expand_dims(
                    jnp.concatenate([states, a], axis=1), axis=1
                )
                state_action_history = jnp.concatenate(
                    [state_action_history[:, 1:, :], state_action], axis=1
                )

                if self.env.use_adaptive_approach is True:
                    ext = self.adaptation_module(state_action_history)
                else:
                    ext = jnp.empty((self.env.num_envs, 0))

                # Check if a timeout is appropriate
                timeout = ep_len == self.max_ep_len
                epoch_ended = t == self.steps_per_epoch - 1

                # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
                # for each env individually
                if terminal.all() or timeout or epoch_ended:
                    self.env.reset()
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

            # Get the data from the training loop
            data = buffer.get()

            obs = data["obs"].reshape(-1, self.env.obs_dim)
            act = data["act"].reshape(-1, self.env.act_dim)
            if self.env.use_adaptive_approach is True:
                extrinsics = data["extrinsics"].reshape(-1, self.env.ext_dim)
            else:
                extrinsics = jnp.empty((self.steps_per_epoch * self.env.num_envs, 0))

            state_action_data = jnp.concatenate([obs, act], axis=1)

            key = jax.random.PRNGKey(42)
            num_epochs = 100

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
            X_train, y_train, X_val, y_val = train_val_split(
                state_action_data, extrinsics
            )

            # Training loop
            for epoch in range(num_epochs):
                # Compute the loss
                am_train_loss, grads = nnx.value_and_grad(mse_loss_fn)(
                    self.am,
                    X_train,
                    y_train,
                    key,
                )
                print(f"{am_train_loss = }\n")
                self.am_optimizer.update(grads)

                # Periodically evaluate on the validation set (e.g., every 10 epochs)
                if epoch % 10 == 0:
                    am_val_loss = mse_loss_fn(self.am, X_val, y_val, key)
                    print(
                        f"Epoch {epoch}: Train Loss = {am_train_loss:.4f}, Val Loss = {am_val_loss:.4f}\n"
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
                    run_name=self.env.run_name,
                    stage="am_training",
                    am_train_loss=am_train_loss,
                    am_val_loss=am_val_loss,
                )

            # Save the adaptation module weights
            save_adaptation_module(self.am, self.ckpt_dir, "adapt_module_state.pkl")

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

        # Vectorize adaptation module
        self.adaptation_module = jax.vmap(self.am)

        returns = jnp.zeros((self.env.num_envs, self.n_evals))

        for eval in range(self.n_evals):
            print(f"Testing policy: episode {eval+1}/{self.n_evals}\n")
            self.env.reset()
            states = self.env.get_states(self.reference_point)
            states_normalized = states

            state_action_history = jnp.zeros(
                (self.env.num_envs, 50, self.env.obs_dim + self.env.act_dim)
            )  # No history in the beginning
            if self.env.use_adaptive_approach is True:
                ext = jnp.ones_like(self.env.mjx_batch.ctrl)  # No control input yet
            else:
                ext = jnp.empty((self.env.num_envs, 0))

            ep_ret = jnp.zeros(self.env.num_envs)
            ep_returns = jnp.zeros((self.env.num_envs, self.episode_len))
            ep_obs = jnp.zeros((self.env.num_envs, self.episode_len, self.env.obs_dim))
            terminal = jnp.zeros(self.env.num_envs, dtype=bool)
            for ep in range(self.episode_len):
                actions = self.agent.get_control_input(
                    "evaluation", jnp.concatenate([states, ext], axis=1)
                )  # Use un-normalized states
                states = self.env.get_states(self.reference_point)
                ep_obs = ep_obs.at[:, ep, :].set(states)
                states_normalized = normalize_obs(states, ep_obs, ep)

                state_action = jnp.expand_dims(
                    jnp.concatenate([states, actions], axis=1), axis=1
                )
                state_action_history = jnp.concatenate(
                    [state_action_history[:, 1:, :], state_action], axis=1
                )

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

                rewards, terminal = self.env.transition(actions, states)
                ep_returns = ep_returns.at[:, ep].set(
                    self.gamma * ep_returns[:, ep - 1] + rewards
                )
                rewards_scaled = scale_rews(rewards, ep_returns, ep)
                ep_ret += rewards_scaled
                if terminal.all():  # Abort if all environments terminated
                    break

            returns = returns.at[:, eval].set(ep_ret)

            # Log key RL metrics
            if self.agent.has_logger:
                self.env.logger.log(
                    self.env.run_id,
                    float(self.env.mjx_batch.time[0]),
                    run_name=self.env.run_name,
                    stage="evaluation",
                    mean_scaled_episodic_returns=ep_ret.mean(),
                    num_terminal=jnp.sum(terminal),
                )

        print(
            f"Average episodic return over all evals and all envs: {jnp.mean(returns)}\n"
        )

    def _generate_experience(self) -> None:
        """
        Roll out an episode where the actions are computed from a PD controller that serves as training data.
        """
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

        # Main training loop
        ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
        ep_obs = jnp.zeros((self.env.num_envs, self.steps_per_epoch, self.env.obs_dim))
        for t in range(self.steps_per_epoch):
            _, v, logp = self.agent.act(
                jnp.concatenate([states, ext], axis=1)
            )  # Use un-normalized states
            self.env.obs = self.env.get_obs()
            a = self.pd_ctrl.get_control_input(self.env)

            r, terminal = self.env.transition(a, states)
            ep_returns = ep_returns.at[:, t].set(self.gamma * ep_returns[:, t - 1] + r)
            r_scaled = scale_rews(r, ep_returns, t)
            ep_ret += r_scaled
            ep_len += 1

            # Log transition
            buffer.store(states, a, r_scaled, v, logp, ext)  # Use un-normalized states

            # Update state
            states = self.env.get_states(self.reference_point)
            ep_obs = ep_obs.at[:, t, :].set(states)
            states_normalized = normalize_obs(states, ep_obs, t)

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

    def _load_rl_hyperparams(self) -> None:
        """
        Load the relevant hyperparams from the config file.
        """
        if isinstance(self.agent, VPG):
            VPG._load_vpg_hyperparams(self)
        elif isinstance(self.agent, PPO):
            PPO._load_ppo_hyperparams(self)
        else:
            raise Exception("Agent has not been implemented.")
        self.episode_len = self.env.env_cfg.control.RL.episode_len
        self.n_evals = self.env.env_cfg.control.RL.n_evals

    def _create_checkpoint_file_names(self) -> None:
        """
        Create the checkpoint file names for the pretraining and training data and states.
        """
        if self.env.use_adaptive_approach:
            self.pretraining_data_file_name = "pretraining_data_adaptive.pkl"
            self.pretraining_state_file_name = "pretraining_state_adaptive.pkl"
            if self.env.use_pretrained:
                self.training_data_file_name = "training_data_adaptive_pretrained.pkl"
                self.training_state_file_name = "training_state_adaptive_pretrained.pkl"
            else:
                self.training_data_file_name = "training_data_adaptive.pkl"
                self.training_state_file_name = "training_state_adaptive.pkl"
        else:
            self.pretraining_data_file_name = "pretraining_data.pkl"
            self.pretraining_state_file_name = "pretraining_state.pkl"
            if self.env.use_pretrained:
                self.training_data_file_name = "training_data_pretrained.pkl"
                self.training_state_file_name = "training_state_pretrained.pkl"
            else:
                self.training_data_file_name = "training_data.pkl"
                self.training_state_file_name = "training_state.pkl"
