from collections.abc import Mapping

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


def _build_actor(mode: str, fusion: str, frozen_nominal_actor: bool = True) -> Actor:
    return Actor(
        obs_dim=3,
        act_dim=2,
        hidden_sizes=[4],
        activation=nnx.tanh,
        res_dim=2,
        act_low=jnp.array([-2.0, 0.0], dtype=jnp.float32),
        act_high=jnp.array([2.0, 1.5], dtype=jnp.float32),
        adaptive_policy_mode=mode,
        context_fusion=fusion,
        residual_scale=0.25,
        frozen_nominal_actor=frozen_nominal_actor,
    )


def test_policy_mode_fusion_combinations_shapes_and_stability() -> None:
    obs_res = jnp.array(
        [[0.1, -0.2, 0.3, 0.4, -0.5], [-0.6, 0.7, -0.8, 0.9, -1.0]],
        dtype=jnp.float32,
    )
    key = jax.random.PRNGKey(0)
    combos = [
        ("direct", "concat"),
        ("direct", "film"),
        ("residual", "concat"),
        ("residual", "film"),
    ]
    for mode, fusion in combos:
        actor = _build_actor(mode, fusion)
        pi, _ = actor.forward(obs_res)
        assert pi.mean().shape == (2, 2)
        actions, logp = actor.sample_action_and_logp(obs_res, key)
        assert actions.shape == (2, 2)
        assert logp.shape == (2,)
        assert jnp.all(jnp.isfinite(actions))
        assert jnp.all(jnp.isfinite(logp))


def test_residual_mode_frozen_nominal_blocks_nominal_gradients() -> None:
    actor = _build_actor("residual", "concat", frozen_nominal_actor=True)
    obs_res = jnp.array([[0.2, -0.1, 0.3, 0.4, -0.2]], dtype=jnp.float32)

    @nnx.grad
    def loss_fn(model: Actor):
        mu, _, _, _, _ = model.pre_action_components(obs_res)
        return jnp.sum(mu**2)

    grads = loss_fn(actor)
    state_grads = nnx.state(grads.state_mu_net)
    delta_grads = nnx.state(grads.delta_mu_net)

    def _leaf_l2_sum(state_tree) -> float:
        total = 0.0

        def _visit(node):
            nonlocal total
            if hasattr(node, "value"):
                total += float(jnp.sum(jnp.square(node.value)))
                return
            if isinstance(node, Mapping):
                for child in node.values():
                    _visit(child)

        _visit(state_tree)
        return total

    state_l2 = _leaf_l2_sum(state_grads)
    delta_l2 = _leaf_l2_sum(delta_grads)
    assert state_l2 == 0.0
    assert delta_l2 > 0.0
