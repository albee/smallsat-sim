import jax
import jax.numpy as jnp
from flax import nnx
import optax

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.controllers.rl.algorithms.base_agent import BaseAgent


class PPOAgent(BaseAgent):
    """
    Base implementation for actor-critic agents.
    """

    def update_policy_gradient(
        self,
        actor_lr: float,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Update the policy gradient.
        """

        # Define the actor loss
        @nnx.jit
        def actor_loss_fn(actor_model, tdres: jnp.ndarray):
            _, logp_a = actor_model.forward(obs, actions)
            return -jnp.sum(tdres * logp_a)

        # Initialize an ADAM optimizers for the actor network
        actor_optimizer = nnx.Optimizer(self.actor, optax.adam(learning_rate=actor_lr))

        # Compute the actor loss
        actor_loss, grads = nnx.value_and_grad(actor_loss_fn)(self.actor, tdres)
        print(f"{actor_loss = }")

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
