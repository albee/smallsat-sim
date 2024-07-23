import jax
import jax.numpy as jnp
from flax import nnx
import tensorflow_probability.substrates.jax as tfp

from smallsat_sim.controllers.rl.modules.mlp import mlp


class Actor(nnx.Module):
    """
    The policy network. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """
    def __init__(self, obs_dim: int, act_dim: int, hidden_sizes: int, activation) -> None:
        super().__init__()
        log_std = -0.5 * jnp.ones(act_dim)
        self.log_std = nnx.Param(log_std) # TODO: double-check this
        self.mu_net = mlp([obs_dim] + list(hidden_sizes) + [act_dim], activation)

    def _distribution(self, obs: jnp.ndarray):
        """
        Return a Gaussian distribution over actions given observations.
        """
        mu = self.mu_net(obs)
        std = jnp.exp(self.log_std)
        return tfp.distributions.MultivariateNormalDiag(mu, std)
    
    def _log_prob_from_dist(self, pi: jnp.ndarray, actions: jnp.ndarray):
        """
        Return the log-probability of actions under the action distribution.
        """
        return pi.log_prob(actions).sum(axis=-1)

    def forward(self, obs: jnp.ndarray, actions=None):
        """
        Return action distributions for given observations and the log-likelihood of given actions under those distributions.
        """
        pi = self._distribution(obs)

        if actions is None:
            logp = None
        else:
            logp = self._log_prob_from_dist(pi, actions)

        return pi, logp
    