import os
import time
import jax
import jax.numpy as jnp
from flax import nnx
import optax
import wandb
import pickle
from typing import Optional

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController
from smallsat_sim.controllers.rl.algorithms.vpg import VPG
from smallsat_sim.controllers.rl.algorithms.ppo import PPO
from smallsat_sim.controllers.rl.storage.replay_buffer import ReplayBuffer
from smallsat_sim.controllers.rl.runners.runner_utils import (
    save_training_data,
    save_trained_modules,
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


class OnPolicyRunner(object):
    """
    On-policy runner for training and evaluation. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """

    def __init__(self, env: VecEnv, planner: BasePlanner) -> None:
        # Initialize the environment and agent
        self.env = env
        self.agent = PPO(self.env, planner)
        self.reference_point = planner.reference_points[0]
        self.pd_ctrl = VectorizedPDController(env, planner)
        self._load_rl_hyperparams()

        # Path to save the checkpoints
        self.ckpt_dir = "smallsat_sim/controllers/rl/checkpoints/"

        # Use Weights and Biases for logging
        if self.env.use_wandb:
            wandb.login()
            wandb.init(
                project="Astrobee-training",
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

    def pretrain(self, strategy: Optional[str] = "supervised_learning") -> None:
        """
        Pretrain the actor and critic networks.
        """
        file_path = os.path.join(self.ckpt_dir, "pretraining_state.pkl")
        if os.path.isfile(file_path):
            return

        print("Pretraining modules...")

        # Generate experience if necessary and load the pretraining data
        self._generate_experience()
        pretraining_data = load_training_data(self.ckpt_dir, "pretraining_data.pkl")

        # Load the data
        obs = pretraining_data["obs"].reshape(-1, self.env.obs_dim)
        act = pretraining_data["act"].reshape(-1, self.env.act_dim)
        ret = pretraining_data["ret"].reshape(-1)
        tdres = pretraining_data["tdres"].reshape(-1)
        logp = pretraining_data["logp"].reshape(-1)

        # Pretrain the policy network
        if strategy == "supervised_learning":
            # Clip the actions to the highest upper bound on the force range of the thrusters
            act_clipped = jnp.where(act > 0.6, 0.6, act)

            # Optimizer to pretrain the policy network
            actor_optimizer = nnx.Optimizer(
                self.agent.actor, optax.adam(learning_rate=1e-2, eps=1e-5)
            )

            # Split into training and validation sets
            X_train, y_train, X_val, y_val = train_val_split(obs, act_clipped)

            actor_losses = []
            actor_val_losses = []
            num_epochs = 80
            num_batches = 256
            batch_size = int(obs.shape[0] / num_batches)
            num_train_samples = X_train.shape[0]
            num_val_samples = X_val.shape[0]

            # Create PRNG keys
            key = jax.random.PRNGKey(42)
            keys = jax.random.split(key, num=num_epochs)

            # Training loop
            for epoch in range(num_epochs):
                # Shuffle the training data
                permutation = jax.random.permutation(keys[epoch], num_train_samples)
                X_train = X_train[permutation]
                y_train = y_train[permutation]

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
                    f"Epoch: {epoch+1:2} avg. actor training loss: {sum(actor_losses)/len(actor_losses)}"
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
                    f"Epoch: {epoch+1:2} avg. actor evaluation loss: {sum(actor_val_losses)/len(actor_val_losses)}"
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
            jax.random.PRNGKey(42), obs, ret, minibatch=False
        )

        # Save the trained actor and critic network weights
        save_trained_modules(self.agent, self.ckpt_dir, "pretraining_state.pkl")

    def learn(self) -> None:
        """
        Main training loop.
        """
        print("Training agent...")

        # Check if pretrained actor and critic modules are available and load them
        file_path = os.path.join(self.ckpt_dir, "pretraining_state.pkl")
        if os.path.isfile(file_path):
            restored_state = load_trained_modules(
                self.ckpt_dir, "pretraining_state.pkl"
            )
            nnx.update(self.agent.actor.mu_net, restored_state["actor_model"].mu_net)
            nnx.update(self.agent.critic.v_net, restored_state["critic_model"].v_net)
        else:
            print("No pretrained modules available.")

        # Set up buffer
        buffer = ReplayBuffer(
            self.env.num_envs,
            self.env.obs_dim,
            self.env.act_dim,
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

        # Create PRNG keys
        key = jax.random.PRNGKey(42)
        keys = jax.random.split(key, num=self.epochs)

        # Main training loop
        for epoch in range(self.epochs):
            ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
            ep_obs = jnp.zeros((self.env.num_envs, self.steps_per_epoch, self.env.obs_dim))
            for t in range(self.steps_per_epoch):
                a, v, logp = self.agent.act(states_normalized, epoch)

                r, terminal = self.agent.env.transition(a, states, epoch)
                ep_returns = ep_returns.at[:, t].set(
                    self.gamma * ep_returns[:, t - 1] + r
                )
                r_scaled = scale_rews(r, ep_returns, t)
                ep_ret += r_scaled
                ep_len += 1

                # Log transition
                buffer.store(states_normalized, a, r_scaled, v, logp)

                # Update state
                states = self.env.get_states(self.reference_point)
                ep_obs = ep_obs.at[:, t, :].set(states)
                states_normalized = normalize_obs(states, ep_obs, t)

                # Check if a timeout is appropriate
                timeout = ep_len == self.max_ep_len
                epoch_ended = t == self.steps_per_epoch - 1

                # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
                # for each env individually
                if terminal.all() or timeout or epoch_ended:
                    # If the trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.agent.act(states_normalized, epoch)
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
                            run_id=self.env.run_id,
                            timestamp=float(self.env.mjx_batch.time[0]),
                            mean_scaled_episodic_returns=ep_ret.mean(),
                        )

                    self.env.reset()
                    states, ep_ret, ep_len = (
                        self.env.get_states(self.reference_point),
                        jnp.zeros(self.env.num_envs),
                        0,
                    )
                    states_normalized = states

            # Get the data from the training loop and save it
            data = buffer.get()
            save_training_data(self.ckpt_dir, "training_data.pkl", data)

            obs = data["obs"].reshape(-1, self.env.obs_dim)
            actions = data["act"].reshape(-1, self.env.act_dim)
            rews = data["rews"].reshape(-1)
            tdres = data["tdres"].reshape(-1)
            returns = data["ret"].reshape(-1)
            logp = data["logp"].reshape(-1)

            # Policy gradient update
            actor_loss = self.agent.update_policy_gradient(
                keys[epoch],
                obs,
                actions,
                tdres,
                logp,
            )

            # Value function updates
            critic_loss = self.agent.update_value_function(keys[epoch], obs, returns)

            # Monitor key RL metrics during training using Weights & Biases
            if self.env.use_wandb:
                wandb.log(
                    {
                        "mean_scaled_rewards": rews.mean(),
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "num_terminal": jnp.sum(terminal),
                        "avg_log_std": self.agent.actor.log_std.value.mean(),
                    }
                )

            # Log key RL metrics
            if self.agent.has_logger:
                self.env.logger.log(
                    run_id=self.env.run_id,
                    timestamp=float(self.env.mjx_batch.time[0]),
                    mean_scaled_rewards=rews.mean(),
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    num_terminal=jnp.sum(terminal),
                    avg_log_std=self.agent.actor.log_std.value.mean(),
                )

            # Save the trained actor and critic network weights
            save_trained_modules(self.agent, self.ckpt_dir, "training_state.pkl")

    def evaluate(self) -> None:
        """
        Evaluate the agent.
        """
        print("Evaluating agent...")

        # Check if trained actor and critic modules are available and load them
        file_path = os.path.join(self.ckpt_dir, "training_state.pkl")
        if os.path.isfile(file_path):
            restored_state = load_trained_modules(self.ckpt_dir, "training_state.pkl")
            nnx.update(self.agent.actor.mu_net, restored_state["actor_model"].mu_net)
            nnx.update(self.agent.critic.v_net, restored_state["critic_model"].v_net)
        else:
            raise Exception("No training has been done yet.")

        returns = jnp.zeros((self.env.num_envs, self.n_evals))

        for eval in range(self.n_evals):
            print(f"Testing policy: episode {eval+1}/{self.n_evals}")
            self.env.reset()
            states = self.env.get_states(self.reference_point)
            states_normalized = states
            ep_ret = jnp.zeros(self.env.num_envs)
            ep_returns = jnp.zeros((self.env.num_envs, self.episode_len))
            ep_obs = jnp.zeros((self.env.num_envs, self.episode_len, self.env.obs_dim))
            terminal = jnp.zeros(self.env.num_envs, dtype=bool)
            for ep in range(self.episode_len):
                actions = self.agent.get_control_input(states_normalized)
                states = self.env.get_states(self.reference_point)
                ep_obs = ep_obs.at[:, ep, :].set(states)
                states_normalized = normalize_obs(states, ep_obs, ep)
                rewards, terminal = self.env.transition(actions, states)
                ep_returns = ep_returns.at[:, ep].set(
                    self.gamma * ep_returns[:, ep - 1] + rewards
                )
                rewards_scaled = scale_rews(rewards, ep_returns, ep)
                ep_ret += rewards_scaled
                if terminal.all():  # Abort if all environments terminated
                    break
                # Log key RL metrics
                if self.agent.has_logger:
                    self.env.logger.log(
                        run_id=self.env.run_id,
                        timestamp=float(self.env.mjx_batch.time[0]),
                        states=states,
                        num_terminal=jnp.sum(terminal),
                    )
            returns = returns.at[:, eval].set(ep_ret)
        print(
            f"Average episodic return over all evals and all envs: {jnp.mean(returns)}"
        )

    def _generate_experience(self) -> None:
        """
        Roll out an episode where the actions are computed from a PD controller that serves as training data.
        """
        file_path = os.path.join(self.ckpt_dir, "pretraining_data.pkl")
        if os.path.isfile(file_path):
            return

        print("Gather training data using the PD controller...")

        # Set up buffer
        buffer = ReplayBuffer(
            self.env.num_envs,
            self.env.obs_dim,
            self.env.act_dim,
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

        # Main training loop
        ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
        ep_obs = jnp.zeros((self.env.num_envs, self.steps_per_epoch, self.env.obs_dim))
        for t in range(self.steps_per_epoch):
            _, v, logp = self.agent.act(states_normalized)
            self.env.obs = self.env.get_obs()
            ctrl_input = self.pd_ctrl.get_control_input(self.env)
            a = jnp.asarray(ctrl_input)

            r, terminal = self.agent.env.transition(a, states)
            ep_returns = ep_returns.at[:, t].set(self.gamma * ep_returns[:, t - 1] + r)
            r_scaled = scale_rews(r, ep_returns, t)
            ep_ret += r_scaled
            ep_len += 1

            # Log transition
            buffer.store(states_normalized, a, r_scaled, v, logp)

            # Update state
            states = self.env.get_states(self.reference_point)
            ep_obs = ep_obs.at[:, t, :].set(states)
            states_normalized = normalize_obs(states, ep_obs, t)

            # Check if a timeout is appropriate
            timeout = ep_len == self.max_ep_len
            epoch_ended = t == self.steps_per_epoch - 1

            # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
            # for each env individually
            if terminal.all() or timeout or epoch_ended:
                # If the trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = self.agent.act(states_normalized)
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

        # Get the data from the training loop and save it
        data = buffer.get()
        save_training_data(self.ckpt_dir, "pretraining_data.pkl", data)

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
