import jax
import numpy as np
import jax.numpy as jnp
from flax import nnx
import optax
from functools import partial

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.controllers.rl.algorithms.base_agent import BaseAgent
from smallsat_sim.utils.helpers_jax import (
    calc_lateral_tracking_error,
    calc_orientation_error,
    calc_attitude_error,
    calc_extrinsic_error,
)


class PPO(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    """

    def __init__(self, env, planner, activation=nnx.tanh) -> None:
        super().__init__(env, planner, activation)

        # Load hyperparams
        self._load_ppo_hyperparams()

        # Set the number of training epochs
        self.actor_training_epochs = 100
        self.critic_training_epochs = 100
        self.actor_critic_training_epochs = 100

        # Total batch size
        self.batch_size = self.env.num_envs * self.steps_per_epoch

        # Define the size of each mini-batch
        self.num_minibatches = 16
        self.minibatch_size = int(np.ceil(self.batch_size / self.num_minibatches))

        # Compute the total number of steps
        actor_total_steps = int(
            self.epochs * self.actor_training_epochs * self.num_minibatches
        )  # Upper bound because of early stopping
        critic_total_steps = int(
            self.epochs * self.critic_training_epochs * self.num_minibatches
        )

        # Initialize an ADAM optimizers for the actor, critic and encoder networks with linear lr decay
        self.actor_optimizer = nnx.Optimizer(
            self.actor,
            optax.adam(
                learning_rate=optax.schedules.linear_schedule(
                    self.actor_lr, self.actor_lr * 0.01, actor_total_steps
                ),
                eps=1e-5,
            ),
        )
        self.critic_optimizer = nnx.Optimizer(
            self.critic,
            optax.adam(
                learning_rate=optax.schedules.linear_schedule(
                    self.critic_lr, self.critic_lr * 0.01, critic_total_steps
                ),
                eps=1e-5,
            ),
        )

        # Set the clip ratio and the target kl divergence
        self.clip_ratio = 0.2
        self.target_kl = 0.01

    # Define the actor loss
    @partial(nnx.jit, static_argnums=(0,))
    def jit_actor_loss_fn(
        self,
        actor_model,
        tdres: jnp.ndarray,
        obs_extrinsics: jnp.ndarray,
        actions: jnp.ndarray,
        logp: jnp.ndarray,
        clip_ratio: float,
    ):
        _, logp_a = actor_model.forward(obs_extrinsics, actions)
        ratio = jnp.exp(logp_a - logp)
        clip_adv = jax.lax.clamp(1 - clip_ratio, ratio, 1 + clip_ratio)
        return -jax.lax.min(ratio * tdres, clip_adv).mean()

    # Define the critic loss
    @partial(nnx.jit, static_argnums=(0,))
    def jit_critic_loss_fn(
        self,
        critic_model,
        returns: jnp.ndarray,
        obs_extrinsics: jnp.ndarray,
    ):
        values = critic_model.forward(obs_extrinsics)
        return jnp.mean((values - returns) ** 2)  # MSE loss

    def update_policy_gradient(
        self,
        key,
        obs_extrinsics: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
        minibatch: bool = True,
    ) -> jnp.ndarray:
        """
        Update the policy gradient.
        """

        # Create PRNG keys
        keys = jax.random.split(key, num=self.actor_training_epochs)

        # Compute the actor loss
        for i in range(self.actor_training_epochs):
            shuffled_indices = jax.random.permutation(
                keys[i], jnp.arange(self.batch_size)
            )
            shuffled_obs_extrinsics = obs_extrinsics[shuffled_indices]
            shuffled_actions = actions[shuffled_indices]
            shuffled_tdres = tdres[shuffled_indices]
            shuffled_logp = logp[shuffled_indices]

            if minibatch:
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size

                    actor_loss, grads = nnx.value_and_grad(self.jit_actor_loss_fn)(
                        self.actor,
                        shuffled_tdres[start:end],
                        shuffled_obs_extrinsics[start:end],
                        shuffled_actions[start:end],
                        shuffled_logp[start:end],
                        self.clip_ratio,
                    )
                    print(f"{actor_loss = }")

                    _, logp_a = self.actor.forward(
                        shuffled_obs_extrinsics[start:end], shuffled_actions[start:end]
                    )

                    kl = (shuffled_logp[start:end] - logp_a).mean()
                    if kl > 1.5 * self.target_kl:
                        print("Early stopping at step %d due to reaching max kl" % i)
                        break

                    # Update the gradients
                    self.actor_optimizer.update(grads)

            else:
                actor_loss, grads = nnx.value_and_grad(self.jit_actor_loss_fn)(
                    self.actor,
                    tdres,
                    obs_extrinsics,
                    actions,
                    logp,
                    self.clip_ratio,
                )
                print(f"{actor_loss = }")

                _, logp_a = self.actor.forward(obs_extrinsics, actions)

                kl = (logp - logp_a).mean()
                if kl > 1.5 * self.target_kl:
                    print("Early stopping at step %d due to reaching max kl" % i)
                    break

                # Update the gradients
                self.actor_optimizer.update(grads)

        return actor_loss

    def update_value_function(
        self,
        key,
        obs_extrinsics: jnp.ndarray,
        returns: jnp.ndarray,
        minibatch: bool = True,
    ) -> jnp.ndarray:
        """
        Update the value function.
        """

        # Create PRNG keys
        keys = jax.random.split(key, num=self.critic_training_epochs)

        for i in range(self.critic_training_epochs):
            # Shuffle the indices
            shuffled_indices = jax.random.permutation(
                keys[i], jnp.arange(self.batch_size)
            )

            if minibatch:
                for start in range(0, self.batch_size, self.minibatch_size):
                    end = start + self.minibatch_size
                    mb_indices = shuffled_indices[start:end]

                    # Compute the critic loss
                    critic_loss, grads = nnx.value_and_grad(self.jit_critic_loss_fn)(
                        self.critic, returns[mb_indices], obs_extrinsics[mb_indices]
                    )
                    print(f"{critic_loss = }")

                    # Update the gradients
                    self.critic_optimizer.update(grads)

            else:
                # Compute the critic loss
                critic_loss, grads = nnx.value_and_grad(self.jit_critic_loss_fn)(
                    self.critic, returns, obs_extrinsics
                )
                print(f"{critic_loss = }")

                # Update the gradients
                self.critic_optimizer.update(grads)

        return critic_loss

    def update_actor_critic_minibatch(
        self,
        key,
        obs_extrinsics: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
        returns: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Update the policy gradient and the value function (both at each minibatch, as opposed to doing it sequentially).
        """
        # Create PRNG keys
        keys = jax.random.split(key, num=self.actor_critic_training_epochs)

        # Compute the actor loss
        for i in range(self.actor_critic_training_epochs):
            shuffled_indices = jax.random.permutation(
                keys[i], jnp.arange(self.batch_size)
            )
            shuffled_obs_extrinsics = obs_extrinsics[shuffled_indices]
            shuffled_actions = actions[shuffled_indices]
            shuffled_tdres = tdres[shuffled_indices]
            shuffled_logp = logp[shuffled_indices]
            shuffled_returns = returns[shuffled_indices]

            for start in range(0, self.batch_size, self.minibatch_size):
                end = start + self.minibatch_size

                actor_loss, grads = nnx.value_and_grad(self.jit_actor_loss_fn)(
                    self.actor,
                    shuffled_tdres[start:end],
                    shuffled_obs_extrinsics[start:end],
                    shuffled_actions[start:end],
                    shuffled_logp[start:end],
                    self.clip_ratio,
                )
                print(f"{actor_loss = }")

                _, logp_a = self.actor.forward(
                    shuffled_obs_extrinsics[start:end], shuffled_actions[start:end]
                )

                kl = (shuffled_logp[start:end] - logp_a).mean()
                if kl > 1.5 * self.target_kl:
                    print("Early stopping at step %d due to reaching max kl" % i)
                    break

                # Update the gradients
                self.actor_optimizer.update(grads)

                # Compute the critic loss
                critic_loss, grads = nnx.value_and_grad(self.jit_critic_loss_fn)(
                    self.critic,
                    shuffled_returns[start:end],
                    shuffled_obs_extrinsics[start:end],
                )
                print(f"{critic_loss = }")

                # Update the gradients
                self.critic_optimizer.update(grads)

        return actor_loss, critic_loss

    def _load_ppo_hyperparams(self) -> None:
        """
        Load PPO-specific hyperparams.
        """
        self.steps_per_epoch = self.env.env_cfg.control.RL.PPO.steps_per_epoch
        self.epochs = self.env.env_cfg.control.RL.PPO.epochs
        self.max_ep_len = self.env.env_cfg.control.RL.PPO.max_ep_len
        self.gamma = self.env.env_cfg.control.RL.PPO.gamma
        self.lam = self.env.env_cfg.control.RL.PPO.lam
        self.actor_lr = self.env.env_cfg.control.RL.PPO.actor_lr
        self.critic_lr = self.env.env_cfg.control.RL.PPO.critic_lr

    def _log(
        self,
        run_id: int,
        timestamp: float,
        stage: str,
        env: VecEnv,
        ctrl_input: jnp.ndarray,
        obs_extrinsics: jnp.ndarray,
    ) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        if self.has_logger:
            tracking_error = calc_lateral_tracking_error(env.get_obs(), self.planner)
            attitude_error = calc_orientation_error(obs_extrinsics)

            if env.ext_dim != 0:
                # If the extrinsics are included in the observations
                extrinsics = obs_extrinsics[:, env.obs_dim :]
            else:
                # If the extrinsics are not included in the observations
                # NOTE: they are still logged but not need to plot them
                extrinsics = jnp.zeros((env.num_envs, env.act_dim))

            extrinsic_error = calc_extrinsic_error(
                extrinsics,
                env.mjx_batch.ctrl
                / (ctrl_input + 1e-8 * jnp.ones_like(env.mjx_batch.ctrl)),
            )
            self.logger.log(
                run_id,
                timestamp,
                run_name=env.run_name,
                stage=stage,
                # obs_extrinsics=obs_extrinsics,
                mean_tracking_error=tracking_error.mean(),
                mean_attitude_error=attitude_error.mean(),
                mean_extrinsic_error=extrinsic_error.mean(),
            )
