import jax
import jax.numpy as jnp
from flax import nnx
import optax

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.controllers.rl.algorithms.base_agent import BaseAgent


class VPG(BaseAgent):
    """
    Vanilla Policy Gradient (with Generalized Advantage Estimation) agent.
    """

    # Define the actor loss
    def actor_loss_fn(
        self, actor_model, tdres: jnp.ndarray, obs: jnp.ndarray, actions: jnp.ndarray
    ):
        _, logp_a = actor_model.forward(obs, actions)
        return -jnp.sum(tdres * logp_a)

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

        # Initialize an ADAM optimizers for the actor network
        actor_optimizer = nnx.Optimizer(self.actor, optax.adam(learning_rate=actor_lr))

        # Compute the actor loss
        actor_loss, grads = nnx.value_and_grad(self.jitted_actor_loss_fn)(
            self.actor, tdres
        )
        print(f"{actor_loss = }")

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
                self.critic, returns
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
