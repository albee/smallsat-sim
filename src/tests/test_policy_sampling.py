import jax
import jax.numpy as jnp
from flax import nnx

from smallsat_sim.controllers.rl.modules.base_policy import Actor


def test_fast_policy_sampler_matches_distribution_logprob() -> None:
    actor = Actor(
        obs_dim=3,
        act_dim=2,
        hidden_sizes=[4],
        activation=nnx.tanh,
        res_dim=1,
        act_low=jnp.array([-2.0, 0.0], dtype=jnp.float32),
        act_high=jnp.array([2.0, 1.5], dtype=jnp.float32),
    )
    states = jnp.array(
        [[0.1, -0.2, 0.3, 0.4], [-0.5, 0.6, -0.7, 0.8]],
        dtype=jnp.float32,
    )
    key = jax.random.PRNGKey(7)

    pi, _ = actor.forward(states)
    mu = actor.mu_net(states)
    log_std = actor.log_std.value
    pre_actions = mu + jnp.exp(log_std) * jax.random.normal(key, mu.shape)
    actions = actor.apply_action_bounds(pre_actions)

    expected_logp = actor._log_prob_from_dist(pi, actions)
    fast_logp = actor._log_prob_diag_gaussian_pre_squash(
        mu,
        log_std,
        pre_actions,
    )

    assert jnp.allclose(fast_logp, expected_logp, atol=1e-5, rtol=1e-5)
