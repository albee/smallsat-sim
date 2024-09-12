import os
import time
import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx
import optax
import wandb
import pickle

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
        self.ckpt_path = "smallsat_sim/controllers/rl/checkpoints/"

        # Use Weights and Biases for logging
        if self.env.use_wandb:
            wandb.login()
            wandb.init(
                project="Astrobee-training",
                config={
                    "num_envs": self.env.num_envs,
                    "steps_per_epoch": self.steps_per_epoch,
                    "epochs": self.epochs,
                    "max_epoch_len": self.max_epoch_len,
                    "gamma": self.gamma,
                    "lam": self.lam,
                    "actor_lr": self.actor_lr,
                    "critic_lr": self.critic_lr,
                    "episode_len": self.episode_len,
                    "n_evals": self.n_evals,
                },
            )

    def pretrain(self) -> None:
        """
        Pretain the actor (and critic) network(s).
        """
        print("Pretraining modules...")

        # Check if data is available and load it
        ckpt_dir = os.listdir(self.ckpt_path)

        if len(ckpt_dir) == 0:
            raise Exception("No experiences have been generated yet.")
        else:
            pretraining_data = load_training_data(
                self.ckpt_path, "pretraining_data.pkl"
            )

        # Pretrain the policy network
        obs = pretraining_data["obs"].reshape(-1, self.env.obs_dim)
        act = pretraining_data["act"].reshape(-1, self.env.act_dim)

        idx = int(obs.shape[0] * 0.8)
        X_train, y_train = obs[:idx, :], act[:idx, :]
        X_val, y_val = obs[idx:, :], act[idx:, :]

        optimizer = nnx.Optimizer(
            self.agent.actor, optax.adam(learning_rate=1e-3)
        )

        # Define the loss function
        def mse_loss_fn(model, X: jnp.ndarray, y: jnp.ndarray):
            y_pred_dist, _ = model.forward(X)
            y_pred = y_pred_dist.sample(
                seed=jax.random.PRNGKey(np.random.randint(0, 9999))
            )
            return jnp.mean((y_pred - y) ** 2)

        losses = []
        val_losses = []

        for epoch in range(100):
            # Train network
            loss, grads = nnx.value_and_grad(mse_loss_fn)(
                self.agent.actor, X_train, y_train
            )
            losses.append(loss)
            optimizer.update(grads)
            print(f"Epoch: {epoch+1:2} avg. training loss: {sum(losses)/len(losses)}")

            # Validate
            val_loss, _ = nnx.value_and_grad(mse_loss_fn)(
                self.agent.actor, X_val, y_val
            )
            val_losses.append(val_loss)
            print(
                f"Epoch: {epoch+1:2} avg. evaluation loss: {sum(val_losses)/len(val_losses)}"
            )

        # Pretrain the base network
        # TODO

    def learn(self) -> None:
        """
        Main training loop.
        """
        print("Training agent...")

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

        # Main training loop
        for epoch in range(self.epochs):
            ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
            for t in range(self.steps_per_epoch):
                a, v, logp = self.agent.act(states, epoch)

                r, terminal = self.agent.env.transition(a, states, epoch)
                ep_ret += r
                ep_len += 1

                # Log transition
                buffer.store(states, a, r, v, logp)

                # Update state
                states = self.env.get_states(self.reference_point)

                # Check if a timeout is appropriate
                timeout = ep_len == self.max_epoch_len
                epoch_ended = t == self.steps_per_epoch - 1

                # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
                # for each env individually
                if terminal.all() or timeout or epoch_ended:
                    # If the trajectory didn't reach terminal state, bootstrap value target
                    if epoch_ended:
                        _, v, _ = self.agent.act(states, epoch)
                    else:
                        v = jnp.zeros(self.env.num_envs)

                    if timeout:
                        ep_returns = ep_returns.at[:, t].set(ep_ret)

                    if terminal.all():
                        terminal_env_indices = jnp.nonzero(terminal)
                        ep_returns = ep_returns.at[terminal_env_indices, t].set(
                            ep_ret[terminal_env_indices]
                        )

                    buffer.end_traj(v)

                    self.env.reset()
                    states, ep_ret, ep_len = (
                        self.env.get_states(self.reference_point),
                        jnp.zeros(self.env.num_envs),
                        0,
                    )

            mean_return = jnp.mean(ep_returns) if len(ep_returns) > 0 else jnp.nan
            print(
                f"Epoch: {epoch+1}/{self.epochs}, mean return across all envs {mean_return}"
            )

            # Get the data from the training loop and save it
            data = buffer.get()
            save_training_data(self.ckpt_path, "training_data.pkl", data)

            obs = data["obs"]
            actions = data["act"]
            tdres = data["tdres"]
            returns = data["ret"]
            logp = data["logp"]

            # Policy gradient update
            actor_loss = self.agent.update_policy_gradient(
                self.actor_lr,
                obs,
                actions,
                tdres,
                logp,
            )

            # Value function updates
            critic_loss = self.agent.update_value_function(self.critic_lr, obs, returns)

            # Monitor key RL metrics during training using Weights & Biases
            if self.env.use_wandb:
                wandb.log(
                    {
                        "mean_return": mean_return,
                        "actor_loss": actor_loss,
                        "critic_loss": critic_loss,
                        "mean_dist2goal": jnp.sqrt(
                            states[:, 0] ** 2 + states[:, 1] ** 2
                        ).mean(),
                        "num_terminal": jnp.sum(terminal),
                    }
                )

            # Log key RL metrics
            if self.agent.has_logger:
                self.env.logger.log(
                    run_id=self.env.run_id,
                    timestamp=float(self.env.mjx_batch.time[0]),
                    mean_return=mean_return,
                    actor_loss=actor_loss,
                    critic_loss=critic_loss,
                    obs=obs,
                    num_terminal=jnp.sum(terminal),
                )

            # Save the trained actor and critic network weights
            save_trained_modules(self.agent, self.ckpt_path, "training_state.pkl")

    def evaluate(self) -> None:
        """
        Evaluate the agent.
        """
        print("Evaluating agent...")

        # Check if trained actor and critic modules are available and load them
        ckpt_dir = os.listdir(self.ckpt_path)

        if len(ckpt_dir) == 0:
            raise Exception("No training has been done yet.")
        else:
            restored_state = load_trained_modules(self.ckpt_path, "training_state.pkl")
            nnx.update(self.agent.actor.mu_net, restored_state["actor_model"].mu_net)
            nnx.update(self.agent.critic.v_net, restored_state["critic_model"].v_net)

        returns = jnp.zeros((self.env.num_envs, self.n_evals))

        for eval in range(self.n_evals):
            print(f"Testing policy: episode {eval+1}/{self.n_evals}")
            states = self.env.get_states(self.reference_point)
            cum_returns = jnp.zeros(self.env.num_envs)
            terminal = jnp.zeros(self.env.num_envs, dtype=bool)
            self.env.reset()
            for _ in range(self.episode_len):
                actions = self.agent.get_control_input(states)
                states = self.env.get_states(self.reference_point)
                rewards, terminal = self.env.transition(actions, states)
                cum_returns += rewards
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
            returns = returns.at[:, eval].set(cum_returns)
        print(f"Average return over all envs: {jnp.mean(cum_returns)}")

    def _generate_experience(self) -> None:
        """
        Roll out an episode where the actions are computed from a PD controller that serves as training data.
        """
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

        # Main training loop
        ep_returns = jnp.zeros((self.env.num_envs, self.steps_per_epoch))
        for t in range(self.steps_per_epoch):
            _, v, logp = self.agent.act(states)
            self.env.obs = self.env.get_obs()
            ctrl_input = self.pd_ctrl.get_control_input(self.env)
            a = jnp.asarray(ctrl_input)

            r, terminal = self.agent.env.transition(a, states)
            ep_ret += r
            ep_len += 1

            # Log transition
            buffer.store(states, a, r, v, logp)

            # Update state
            states = self.env.get_states(self.reference_point)

            # Check if a timeout is appropriate
            timeout = ep_len == self.max_epoch_len
            epoch_ended = t == self.steps_per_epoch - 1

            # N. B.: could also have a different ep_len for each env and consider timeout and terminal conditions
            # for each env individually
            if terminal.all() or timeout or epoch_ended:
                # If the trajectory didn't reach terminal state, bootstrap value target
                if epoch_ended:
                    _, v, _ = self.agent.act(states)
                else:
                    v = jnp.zeros(self.env.num_envs)

                if timeout:
                    ep_returns = ep_returns.at[:, t].set(ep_ret)

                if terminal.all():
                    terminal_env_indices = jnp.nonzero(terminal)
                    ep_returns = ep_returns.at[terminal_env_indices, t].set(
                        ep_ret[terminal_env_indices]
                    )

                buffer.end_traj(v)

                self.env.reset()
                states, ep_ret, ep_len = (
                    self.env.get_states(self.reference_point),
                    jnp.zeros(self.env.num_envs),
                    0,
                )

        # Get the data from the training loop and save it
        data = buffer.get()
        save_training_data(self.ckpt_path, "pretraining_data.pkl", data)

    def _load_rl_hyperparams(self) -> None:
        """
        Load the relevant hyperparams from the config file.
        """
        if isinstance(self.agent, VPG):
            self.steps_per_epoch = self.env.env_cfg.control.RL.VPG.steps_per_epoch
            self.epochs = self.env.env_cfg.control.RL.VPG.epochs
            self.max_epoch_len = self.env.env_cfg.control.RL.VPG.max_epoch_len
            self.gamma = self.env.env_cfg.control.RL.VPG.gamma
            self.lam = self.env.env_cfg.control.RL.VPG.lam
            self.actor_lr = self.env.env_cfg.control.RL.VPG.actor_lr
            self.critic_lr = self.env.env_cfg.control.RL.VPG.critic_lr
        elif isinstance(self.agent, PPO):
            self.steps_per_epoch = self.env.env_cfg.control.RL.PPO.steps_per_epoch
            self.epochs = self.env.env_cfg.control.RL.PPO.epochs
            self.max_epoch_len = self.env.env_cfg.control.RL.PPO.max_epoch_len
            self.gamma = self.env.env_cfg.control.RL.PPO.gamma
            self.lam = self.env.env_cfg.control.RL.PPO.lam
            self.actor_lr = self.env.env_cfg.control.RL.PPO.actor_lr
            self.critic_lr = self.env.env_cfg.control.RL.PPO.critic_lr
        else:
            raise Exception("Agent has not been implemented.")
        self.episode_len = self.env.env_cfg.control.RL.episode_len
        self.n_evals = self.env.env_cfg.control.RL.n_evals
