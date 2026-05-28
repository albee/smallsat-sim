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
        initial_log_std: float = -0.5,
        log_std_min: float | None = None,
        adaptive_policy_mode: str = "direct",
        context_fusion: str = "concat",
        residual_scale: float = 1.0,
        frozen_nominal_actor: bool = True,
    ) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        self.act_dim = act_dim
        self.res_dim = res_dim
        self.adaptive_policy_mode = str(adaptive_policy_mode)
        self.context_fusion = str(context_fusion)
        self.residual_scale = float(residual_scale)
        self.frozen_nominal_actor = bool(frozen_nominal_actor)
        if self.adaptive_policy_mode not in ("direct", "residual"):
            raise ValueError("adaptive_policy_mode must be 'direct' or 'residual'.")
        if self.context_fusion not in ("concat", "film"):
            raise ValueError("context_fusion must be 'concat' or 'film'.")
        self._eps = 1e-6
        self.act_low = jnp.asarray(act_low)
        self.act_high = jnp.asarray(act_high)
        self.act_range = self.act_high - self.act_low
        self._range_mask = (self.act_range > 0).astype(jnp.float32)
        self._range_safe = jnp.where(
            self._range_mask > 0, self.act_range, jnp.ones_like(self.act_range)
        )
        self._log_range_safe = jnp.log(self._range_safe)
        log_std = initial_log_std * jnp.ones(act_dim)
        self.log_std = nnx.Param(log_std)
        if log_std_min is None:
            self.log_std_min = None
        else:
            self.log_std_min = jnp.asarray(log_std_min, dtype=jnp.float32)
        if isinstance(hidden_sizes, int):
            hidden_layer_sizes = [hidden_sizes]
        else:
            hidden_layer_sizes = list(hidden_sizes)
        layer_sizes_concat = [obs_dim + res_dim] + hidden_layer_sizes + [act_dim]
        self.mu_net = mlp(
            layer_sizes_concat, activation, output_activation=None, last_layer_std=0.01
        )
        self.state_mu_net = mlp(
            [obs_dim] + hidden_layer_sizes + [act_dim],
            activation,
            output_activation=None,
            last_layer_std=0.01,
        )
        self.delta_mu_net = mlp(
            layer_sizes_concat, activation, output_activation=None, last_layer_std=0.01
        )
        self.film_state_net = mlp(
            [obs_dim] + hidden_layer_sizes,
            activation,
            output_activation=None,
            last_layer_std=1.0,
        )
        hidden_dim = int(hidden_layer_sizes[-1]) if hidden_layer_sizes else int(obs_dim)
        self.film_gamma = nnx.Linear(
            res_dim,
            hidden_dim,
            kernel_init=nnx.initializers.constant(1e-4),
            bias_init=nnx.initializers.constant(0.0),
            rngs=nnx.Rngs(params=0),
        )
        self.film_beta = nnx.Linear(
            res_dim,
            hidden_dim,
            kernel_init=nnx.initializers.constant(1e-4),
            bias_init=nnx.initializers.constant(0.0),
            rngs=nnx.Rngs(params=0),
        )
        self.film_out = nnx.Linear(
            hidden_dim,
            act_dim,
            kernel_init=nnx.initializers.constant(0.01),
            bias_init=nnx.initializers.constant(0.0),
            rngs=nnx.Rngs(params=0),
        )
        self.delta_film_state_net = mlp(
            [obs_dim] + hidden_layer_sizes,
            activation,
            output_activation=None,
            last_layer_std=1.0,
        )
        self.delta_film_gamma = nnx.Linear(
            res_dim,
            hidden_dim,
            kernel_init=nnx.initializers.constant(1e-4),
            bias_init=nnx.initializers.constant(0.0),
            rngs=nnx.Rngs(params=0),
        )
        self.delta_film_beta = nnx.Linear(
            res_dim,
            hidden_dim,
            kernel_init=nnx.initializers.constant(1e-4),
            bias_init=nnx.initializers.constant(0.0),
            rngs=nnx.Rngs(params=0),
        )
        self.delta_film_out = nnx.Linear(
            hidden_dim,
            act_dim,
            kernel_init=nnx.initializers.constant(0.01),
            bias_init=nnx.initializers.constant(0.0),
            rngs=nnx.Rngs(params=0),
        )

    def _split_obs_context(self, obs_residuals: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
        if self.res_dim <= 0:
            return obs_residuals[:, : self.obs_dim], jnp.zeros((obs_residuals.shape[0], 0), dtype=obs_residuals.dtype)
        return obs_residuals[:, : self.obs_dim], obs_residuals[:, self.obs_dim : self.obs_dim + self.res_dim]

    def _normalize_context(self, z: jnp.ndarray) -> jnp.ndarray:
        if z.shape[-1] == 0:
            return z
        return jnp.clip(z, -10.0, 10.0)

    def _film_pre_action(self, state: jnp.ndarray, z: jnp.ndarray, *, delta: bool) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        z_norm = self._normalize_context(z)
        if delta:
            h = self.delta_film_state_net(state)
            gamma = self.delta_film_gamma(z_norm)
            beta = self.delta_film_beta(z_norm)
            h_mod = h * (1.0 + gamma) + beta
            pre = self.delta_film_out(h_mod)
            return pre, gamma, beta
        h = self.film_state_net(state)
        gamma = self.film_gamma(z_norm)
        beta = self.film_beta(z_norm)
        h_mod = h * (1.0 + gamma) + beta
        pre = self.film_out(h_mod)
        return pre, gamma, beta

    def pre_action_components(self, obs_residuals: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        state, z = self._split_obs_context(obs_residuals)
        zeros = jnp.zeros((obs_residuals.shape[0], self.act_dim), dtype=obs_residuals.dtype)
        gamma = jnp.zeros((obs_residuals.shape[0], 0), dtype=obs_residuals.dtype)
        beta = jnp.zeros((obs_residuals.shape[0], 0), dtype=obs_residuals.dtype)
        if self.adaptive_policy_mode == "direct":
            if self.context_fusion == "concat":
                return self.mu_net(obs_residuals), zeros, zeros, gamma, beta
            pre, gamma, beta = self._film_pre_action(state, z, delta=False)
            return pre, zeros, zeros, gamma, beta
        nominal_raw = self.state_mu_net(state)
        nominal = (
            jax.lax.stop_gradient(nominal_raw)
            if self.frozen_nominal_actor
            else nominal_raw
        )
        if self.context_fusion == "concat":
            delta = self.delta_mu_net(obs_residuals)
        else:
            delta, gamma, beta = self._film_pre_action(state, z, delta=True)
        final = nominal + self.residual_scale * delta
        return final, nominal, delta, gamma, beta

    def _distribution(self, obs_residuals: jnp.ndarray):
        """
        Return a Gaussian distribution over actions given observations.
        """
        mu, _, _, _, _ = self.pre_action_components(obs_residuals)
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
        mu, _, _, _, _ = self.pre_action_components(obs_residuals)
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
        mu, _, _, _, _ = self.pre_action_components(obs_residuals)
        return self.apply_action_bounds(mu)

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
