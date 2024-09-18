import jax
import jax.numpy as jnp
from flax import nnx
import optax
from functools import partial

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.controllers.rl.algorithms.base_agent import BaseAgent


class VPG(BaseAgent):
    """
    Vanilla Policy Gradient (with Generalized Advantage Estimation) agent.
    """

    def __init__(self, env, planner, activation=nnx.tanh) -> None:
        super().__init__(env, planner, activation)

        # Load the hyperparams
        self._load_vpg_hyperparams()

        # Initialize an ADAM optimizers for the actor and critic networks
        self.actor_optimizer = nnx.Optimizer(
            self.actor, optax.adam(learning_rate=self.actor_lr, eps=1e-5)
        )
        self.critic_optimizer = nnx.Optimizer(
            self.critic, optax.adam(learning_rate=self.critic_lr, eps=1e-5)
        )

    # Define the actor loss
    @partial(nnx.jit, static_argnums=(0,))
    def jit_actor_loss_fn(
        self, actor_model, tdres: jnp.ndarray, obs: jnp.ndarray, actions: jnp.ndarray
    ):
        _, logp_a = actor_model.forward(obs, actions)
        return -jnp.sum(tdres.reshape(-1) * logp_a)

    # Define the critic loss
    @partial(nnx.jit, static_argnums=(0,))
    def jit_critic_loss_fn(self, critic_model, returns: jnp.ndarray, obs: jnp.ndarray):
        values = critic_model.forward(obs)
        return jnp.mean((values - returns.reshape(-1)) ** 2)  # MSE loss

    def update_policy_gradient(
        self,
        key,
        obs: jnp.ndarray,
        actions: jnp.ndarray,
        tdres: jnp.ndarray,
        logp: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Update the policy gradient.
        """

        # Compute the actor loss
        actor_loss, grads = nnx.value_and_grad(self.jit_actor_loss_fn)(
            self.actor, tdres
        )
        print(f"{actor_loss = }")

        # Update the gradients
        self.actor_optimizer.update(grads)

        return actor_loss

    def update_value_function(
        self, key, obs: jnp.ndarray, returns: jnp.ndarray
    ) -> jnp.ndarray:
        """
        Update the value function.
        """

        for _ in range(100):
            # Compute the critic loss
            critic_loss, grads = nnx.value_and_grad(self.jit_critic_loss_fn)(
                self.critic, returns
            )
            print(f"{critic_loss = }")

            # Update the gradients
            self.critic_optimizer.update(grads)

        return critic_loss
    
    def _load_vpg_hyperparams(self) -> None:
        """
        Load VPG-specific hyperparams.
        """
        self.steps_per_epoch = self.env.env_cfg.control.RL.VPG.steps_per_epoch
        self.epochs = self.env.env_cfg.control.RL.VPG.epochs
        self.max_epoch_len = self.env.env_cfg.control.RL.VPG.max_epoch_len
        self.gamma = self.env.env_cfg.control.RL.VPG.gamma
        self.lam = self.env.env_cfg.control.RL.VPG.lam
        self.actor_lr = self.env.env_cfg.control.RL.VPG.actor_lr
        self.critic_lr = self.env.env_cfg.control.RL.VPG.critic_lr

    def _log(self, run_id: int, timestamp: float, env: BaseEnv) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        raise NotImplementedError(
            f"The _log method is not implemented for the class {self.__class__.__name__}"
        )
