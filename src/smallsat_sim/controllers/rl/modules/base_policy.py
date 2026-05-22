from collections.abc import Sequence
import jax
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
        hidden_sizes: Sequence[int],
        activation,
        res_dim: int,
        act_low: jnp.ndarray,
        act_high: jnp.ndarray,
        log_std_min: float | None = None,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self._eps = 1e-6
        self.act_low = jnp.asarray(act_low)
        self.act_high = jnp.asarray(act_high)
        self.act_range = self.act_high - self.act_low
        self._range_mask = (self.act_range > 0).astype(jnp.float32)
        self._range_safe = jnp.where(
            self._range_mask > 0, self.act_range, jnp.ones_like(self.act_range)
        )
        self._log_range_safe = jnp.log(self._range_safe)
        log_std = -0.5 * jnp.ones(act_dim)
        self.log_std = nnx.Param(log_std)
        if log_std_min is None:
            self.log_std_min = None
        else:
            self.log_std_min = jnp.asarray(log_std_min, dtype=jnp.float32)
        if isinstance(hidden_sizes, int):
            hidden_layer_sizes = [hidden_sizes]
        else:
            hidden_layer_sizes = list(hidden_sizes)
        layer_sizes = [obs_dim + res_dim] + hidden_layer_sizes + [act_dim]
        self.mu_net = mlp(
            layer_sizes,
            activation,
            output_activation=None,
            last_layer_std=0.01,
        )

    def _distribution(self, obs_residuals: jnp.ndarray):
        """
        Return a Gaussian distribution over actions given observations.
        """
        mu = self.mu_net(obs_residuals)
        log_std = self.log_std.value
        if self.log_std_min is not None:
            log_std = jnp.maximum(log_std, self.log_std_min)
        std = jnp.exp(log_std)
        return distrax.MultivariateNormalDiag(mu, std)

    def sample_action_and_logp(
        self, obs_residuals: jnp.ndarray, key: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Fast rollout sampler for the diagonal Gaussian policy.

        PPO rollouts call this for every environment at every step, so avoid
        constructing a Distrax distribution object in the scan hot path.
        """
        mu = self.mu_net(obs_residuals)
        log_std = self.log_std.value
        if self.log_std_min is not None:
            log_std = jnp.maximum(log_std, self.log_std_min)
        std = jnp.exp(log_std)
        noise = jax.random.normal(key, mu.shape, dtype=mu.dtype)
        pre_actions = mu + std * noise
        actions = self.apply_action_bounds(pre_actions)
        logp = self._log_prob_diag_gaussian_pre_squash(mu, log_std, pre_actions)
        return actions, logp

    def _log_prob_from_dist(
        self, pi: distrax.MultivariateNormalDiag, actions: jnp.ndarray
    ):
        """
        Return the log-probability of actions under the action distribution.
        """
        if self.act_dim == 0:
            return pi.log_prob(actions)

        pre_actions = self._inverse_squash(actions)
        norm_actions = self._normalize_actions(actions)
        log_det = self._log_det_jacobian(norm_actions)
        return pi.log_prob(pre_actions) - log_det

    def _log_prob_from_pre_squash(
        self, pi: distrax.MultivariateNormalDiag, pre_actions: jnp.ndarray
    ):
        """
        Log-probability for freshly sampled bounded actions.

        This avoids inverting the sigmoid squash when the pre-squash sample is
        already available in the PPO rollout hot path.
        """
        if self.act_dim == 0:
            return pi.log_prob(pre_actions)

        log_sigmoid = jax.nn.log_sigmoid(pre_actions)
        log_one_minus_sigmoid = jax.nn.log_sigmoid(-pre_actions)
        contrib = self._range_mask * (
            self._log_range_safe + log_sigmoid + log_one_minus_sigmoid
        )
        log_det = jnp.sum(contrib, axis=-1)
        return pi.log_prob(pre_actions) - log_det

    def _log_prob_diag_gaussian_pre_squash(
        self,
        mu: jnp.ndarray,
        log_std: jnp.ndarray,
        pre_actions: jnp.ndarray,
    ) -> jnp.ndarray:
        """
        Corrected log-probability for a diagonal Gaussian followed by the
        sigmoid/range action transform.
        """
        if self.act_dim == 0:
            centered = pre_actions - mu
            inv_std = jnp.exp(-log_std)
            normal_terms = (centered * inv_std) ** 2 + 2.0 * log_std + jnp.log(
                2.0 * jnp.pi
            )
            return -0.5 * jnp.sum(normal_terms, axis=-1)

        centered = pre_actions - mu
        inv_std = jnp.exp(-log_std)
        normal_terms = (centered * inv_std) ** 2 + 2.0 * log_std + jnp.log(
            2.0 * jnp.pi
        )
        logp_pre_squash = -0.5 * jnp.sum(normal_terms, axis=-1)

        log_sigmoid = jax.nn.log_sigmoid(pre_actions)
        log_one_minus_sigmoid = jax.nn.log_sigmoid(-pre_actions)
        transform_terms = self._range_mask * (
            self._log_range_safe + log_sigmoid + log_one_minus_sigmoid
        )
        return logp_pre_squash - jnp.sum(transform_terms, axis=-1)

    def forward(
        self, obs_residuals: jnp.ndarray, actions: jnp.ndarray | None = None
    ) -> tuple[distrax.MultivariateNormalDiag, jnp.ndarray | None]:
        """
        Return action distributions for given observations and the log-likelihood of given actions under those distributions.
        """
        pi = self._distribution(obs_residuals)

        if actions is None:
            logp = None
        else:
            logp = jnp.asarray(self._log_prob_from_dist(pi, actions))

        return pi, logp

    def apply_action_bounds(self, pre_actions: jnp.ndarray) -> jnp.ndarray:
        """
        Squash pre-activation actions into the valid thruster range.
        """
        if self.act_dim == 0:
            return pre_actions
        squashed = jax.nn.sigmoid(pre_actions)
        return self.act_low + self.act_range * squashed

    def deterministic_action(self, obs_residuals: jnp.ndarray) -> jnp.ndarray:
        """
        Return the mean action squashed into the thruster range.
        """
        return self.apply_action_bounds(self.mu_net(obs_residuals))

    def _normalize_actions(self, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Normalize bounded actions to (0, 1) for change-of-variables calculations.
        """
        if self.act_dim == 0:
            return actions
        norm = (actions - self.act_low) / self._range_safe
        norm = jnp.where(self._range_mask > 0, norm, 0.5)
        return jnp.clip(norm, self._eps, 1.0 - self._eps)

    def _inverse_squash(self, actions: jnp.ndarray) -> jnp.ndarray:
        """
        Map bounded actions back to the unconstrained space.
        """
        if self.act_dim == 0:
            return actions
        norm = self._normalize_actions(actions)
        return jnp.log(norm) - jnp.log1p(-norm)

    def _log_det_jacobian(self, norm_actions: jnp.ndarray) -> jnp.ndarray:
        """
        Compute log-determinant of the sigmoid+scaling Jacobian for log-prob correction.
        """
        if self.act_dim == 0:
            return jnp.zeros(norm_actions.shape[0])
        log_sigma = jnp.log(norm_actions)
        log_one_minus = jnp.log1p(-norm_actions)
        contrib = self._range_mask * (self._log_range_safe + log_sigma + log_one_minus)
        return jnp.sum(contrib, axis=-1)
