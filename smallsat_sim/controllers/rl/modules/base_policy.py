import jax.numpy as jnp
from flax import nnx
import distrax

from smallsat_sim.controllers.rl.modules.mlp import mlp


class Actor(nnx.Module):
    """
    The policy network. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """

    def __init__(
        self,
        obs_dim: int,
        act_dim: int,
        hidden_sizes: int,
        activation,
        ext_dim: int,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        log_std = -2.0 * jnp.ones(act_dim)
        self.log_std = nnx.Param(log_std)
        self.mu_net = mlp(
            [obs_dim + ext_dim] + list(hidden_sizes) + [act_dim],
            activation,
            output_activation=nnx.relu,  # ReLU because the control inputs must be positive
            last_layer_std=0.01,
        )

    def _distribution(self, obs_extrinsics: jnp.ndarray):
        """
        Return a Gaussian distribution over actions given observations.
        """
        mu = self.mu_net(obs_extrinsics)
        std = jnp.exp(self.log_std.value)
        return distrax.MultivariateNormalDiag(mu, std)

    def _log_prob_from_dist(self, pi: jnp.ndarray, actions: jnp.ndarray):
        """
        Return the log-probability of actions under the action distribution.
        """
        return pi.log_prob(actions)

    def forward(self, obs_extrinsics: jnp.ndarray, actions: jnp.ndarray | None = None):
        """
        Return action distributions for given observations and the log-likelihood of given actions under those distributions.
        """
        pi = self._distribution(obs_extrinsics)

        if actions is None:
            logp = None
        else:
            logp = self._log_prob_from_dist(pi, actions)

        return pi, logp
