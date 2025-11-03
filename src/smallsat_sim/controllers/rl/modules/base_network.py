import jax.numpy as jnp
from flax import nnx

from smallsat_sim.controllers.rl.modules.mlp import mlp


class Critic(nnx.Module):
    """
    The network used by the value function.
    """

    def __init__(
        self,
        obs_dim: int,
        hidden_sizes: int,
        activation,
        res_dim: int,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.v_net = mlp(
            [obs_dim + res_dim] + [hidden_sizes] + [1],
            activation,
            last_layer_std=1.0,
        )

    def forward(self, obs_residuals: jnp.ndarray):
        """
        Return the value estimates for given observations.
        """
        return jnp.squeeze(self.v_net(obs_residuals), -1)
