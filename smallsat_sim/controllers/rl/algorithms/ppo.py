import jax
import jax.numpy as jnp
from flax import nnx
import optax

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.controllers.rl.algorithms.base_agent import BaseAgent


class PPO(BaseAgent):
    """
    Proximal Policy Optimization (PPO) agent.
    """

    # Define the actor loss
    def actor_loss_fn(
        self,
        actor_model,
        tdres: jnp.ndarray,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        logp: jnp.ndarray,
        clip_ratio: float,
    ):
        _, logp_a = actor_model.forward(obs, actions)
        ratio = jnp.exp(logp_a - logp)
        clip_adv = jax.lax.clamp(1 - clip_ratio, ratio, 1 + clip_ratio)
        return -jax.lax.min(ratio * tdres, clip_adv).mean()

    jitted_actor_loss_fn = nnx.jit(actor_loss_fn, static_argnums=(0,))

    # Define the critic loss
    def critic_loss_fn(self, critic_model, returns: jnp.ndarray, obs: jnp.ndarray):
        values = critic_model.forward(obs)
        return jnp.mean((values - returns) ** 2)  # MSE loss

    jitted_critic_loss_fn = nnx.jit(critic_loss_fn, static_argnums=(0,))

    def update_policy_gradient(
        self,
        actor_lr: float,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Update the policy gradient.
        """

        # Set the clip ratio and the target kl divergence
        clip_ratio = 0.2
        target_kl = 0.01

        # Initialize an ADAM optimizers for the actor network
        actor_optimizer = nnx.Optimizer(self.actor, optax.adam(learning_rate=actor_lr))

        # Compute the actor loss
        for i in range(100):
            actor_loss, grads = nnx.value_and_grad(self.jitted_actor_loss_fn)(
                self.actor, tdres, obs, actions, logp, clip_ratio
            )
            print(f"{actor_loss = }")

            _, logp_a = self.actor.forward(obs, actions)

            kl = (logp - logp_a).mean()
            if kl > 1.5 * target_kl:
                print("Early stopping at step %d due to reaching max kl" % i)
                break

            # Update the gradients
            actor_optimizer.update(grads)

        return actor_loss

    def update_value_function(
        self, critic_lr: float, obs: jnp.ndarray, returns: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Update the value function.
        """

        # Initialize an ADAM optimizer for the critic network
        critic_optimizer = nnx.Optimizer(
            self.critic, optax.adam(learning_rate=critic_lr)
        )

        for _ in range(100):
            # Compute the critic loss
            critic_loss, grads = nnx.value_and_grad(self.jitted_critic_loss_fn)(
                self.critic, returns, obs
            )
            print(f"{critic_loss = }")

            # Update the gradients
            critic_optimizer.update(grads)

        return critic_loss

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        raise NotImplementedError(
            f"The _log method is not implemented for the class {self.__class__.__name__}"
        )
