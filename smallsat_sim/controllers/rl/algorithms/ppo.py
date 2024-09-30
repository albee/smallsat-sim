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

        # Define the actor loss
        # @nnx.jit
        def actor_loss_fn(actor_model, tdres: jnp.ndarray):
            _, logp_a = actor_model.forward(obs, actions)
            ratio = jnp.exp(logp_a - logp)
            clip_adv = jax.lax.clamp(1 - clip_ratio, ratio, 1 + clip_ratio)
            return -jax.lax.min(ratio * tdres, clip_adv).mean()

        # Initialize an ADAM optimizers for the actor network
        actor_optimizer = nnx.Optimizer(self.actor, optax.adam(learning_rate=actor_lr))

        # Compute the actor loss
        for i in range(100):
            actor_loss, grads = nnx.value_and_grad(actor_loss_fn)(self.actor, tdres)
            print(f"{actor_loss = }")

            _, logp_a = self.actor.forward(obs, actions)

            kl = (logp - logp_a).mean()
            if kl > 1.5 * target_kl:
                print("Early stopping at step %d due to reaching max kl" % i)
                break

            # Update the gradients
            actor_optimizer.update(grads)

    def update_value_function(
        self, critic_lr: float, obs: jnp.ndarray, returns: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Update the value function.
        """

        # Define the critic loss
        @nnx.jit
        def critic_loss_fn(critic_model, returns: jnp.ndarray):
            values = critic_model.forward(obs)
            return jnp.mean((values - returns) ** 2)  # MSE loss

        # Initialize an ADAM optimizer for the critic network
        critic_optimizer = nnx.Optimizer(
            self.critic, optax.adam(learning_rate=critic_lr)
        )

        for _ in range(100):
            # Compute the critic loss
            critic_loss, grads = nnx.value_and_grad(critic_loss_fn)(
                self.critic, returns
            )
            print(f"{critic_loss = }")

            # Update the gradients
            critic_optimizer.update(grads)

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        raise NotImplementedError(
            f"The _log method is not implemented for the class {self.__class__.__name__}"
        )
