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
        ext_dim: int = 0,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.v_net = mlp(
            [obs_dim + ext_dim] + list(hidden_sizes) + [1],
            activation,
            last_layer_std=1.0,
        )

    def forward(self, obs_extrinsics: jnp.ndarray):
        """
        Return the value estimates for given observations.
        """
        return jnp.squeeze(self.v_net(obs_extrinsics), -1)
