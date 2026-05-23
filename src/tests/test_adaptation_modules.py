import jax
import jax.numpy as jnp
import pytest

from smallsat_sim.controllers.rl.modules.am_cnn import CNNAdaptationModule
from smallsat_sim.controllers.rl.modules.am_transformer import (
    CrossAttentionAdaptationModule,
    TransformerAdaptationModule,
)


def test_transformer_adaptation_module_output_shapes() -> None:
    module = TransformerAdaptationModule(
        n_steps=8,
        state_action_dim=6,
        ext_dim=3,
        d_model=32,
        n_heads=4,
        mlp_dim=64,
        n_layers=2,
        dropout_rate=0.1,
    )
    history = jnp.ones((8, 6), dtype=jnp.float32)

    pred = module(history)
    mu, log_sigma = module(history, return_stats=True)

    assert pred.shape == (3,)
    assert mu.shape == (3,)
    assert log_sigma.shape == (3,)


def test_transformer_task_query_and_prediction_heads() -> None:
    module = TransformerAdaptationModule(
        n_steps=8,
        state_action_dim=6,
        ext_dim=3,
        query_dim=5,
        predict_delta_dim=4,
        predict_tracking=True,
        d_model=32,
        n_heads=4,
        mlp_dim=64,
        n_layers=2,
        dropout_rate=0.1,
    )
    history = jnp.ones((8, 6), dtype=jnp.float32)
    query = jnp.ones((5,), dtype=jnp.float32)

    mu, log_sigma, delta, tracking = module(
        history,
        query,
        return_stats=True,
        return_predictions=True,
    )

    assert mu.shape == (3,)
    assert log_sigma.shape == (3,)
    assert delta.shape == (4,)
    assert tracking.shape == (1,)


def test_cross_attention_task_query_prediction_heads_and_attention() -> None:
    module = CrossAttentionAdaptationModule(
        n_steps=8,
        state_action_dim=6,
        ext_dim=3,
        query_dim=5,
        predict_delta_dim=4,
        predict_tracking=True,
        d_model=32,
        n_heads=4,
        mlp_dim=64,
        n_layers=2,
        dropout_rate=0.1,
    )
    history = jnp.ones((8, 6), dtype=jnp.float32)
    query = jnp.ones((5,), dtype=jnp.float32)

    mu, log_sigma, delta, tracking, attention = module(
        history,
        query,
        return_stats=True,
        return_predictions=True,
        return_attention=True,
    )
    vmapped = jax.vmap(lambda hist, query_i: module(hist, query_i))(
        jnp.ones((3, 8, 6), dtype=jnp.float32),
        jnp.ones((3, 5), dtype=jnp.float32),
    )

    assert mu.shape == (3,)
    assert log_sigma.shape == (3,)
    assert delta.shape == (4,)
    assert tracking.shape == (1,)
    assert attention.shape == (4, 8)
    assert jnp.allclose(attention.sum(axis=-1), 1.0, atol=1e-5)
    assert vmapped.shape == (3, 3)


def test_transformer_training_flag_controls_dropout() -> None:
    module = TransformerAdaptationModule(
        n_steps=8,
        state_action_dim=6,
        ext_dim=3,
        d_model=32,
        n_heads=4,
        mlp_dim=64,
        n_layers=2,
        dropout_rate=0.9,
    )
    history = jnp.arange(48, dtype=jnp.float32).reshape(8, 6)

    eval_pred_1 = module(history, training=False)
    eval_pred_2 = module(history, training=False)
    train_pred_1 = module(history, training=True)
    train_pred_2 = module(history, training=True)

    assert jnp.allclose(eval_pred_1, eval_pred_2)
    assert not jnp.allclose(train_pred_1, train_pred_2)


def test_cnn_adaptation_module_output_shape_and_min_dim() -> None:
    module = CNNAdaptationModule(n_steps=16, state_action_dim=6, ext_dim=0)
    history = jnp.ones((16, 6), dtype=jnp.float32)

    pred = module(history)

    assert pred.shape == (1,)


def test_cnn_task_query_and_prediction_heads() -> None:
    module = CNNAdaptationModule(
        n_steps=16,
        state_action_dim=6,
        ext_dim=3,
        query_dim=5,
        predict_delta_dim=4,
        predict_tracking=True,
    )
    history = jnp.ones((16, 6), dtype=jnp.float32)
    query = jnp.ones((5,), dtype=jnp.float32)

    context, delta, tracking = module(history, query, return_predictions=True)

    assert context.shape == (3,)
    assert delta.shape == (4,)
    assert tracking.shape == (1,)


def test_transformer_rejects_non_2d_history() -> None:
    module = TransformerAdaptationModule(n_steps=8, state_action_dim=6, ext_dim=3)
    bad_history = jnp.ones((1, 8, 6), dtype=jnp.float32)

    with pytest.raises(ValueError, match="history must be a 2D array"):
        module(bad_history)


def test_transformer_rejects_wrong_step_count() -> None:
    module = TransformerAdaptationModule(n_steps=8, state_action_dim=6, ext_dim=3)
    bad_history = jnp.ones((7, 6), dtype=jnp.float32)

    with pytest.raises(ValueError, match="expected 8 steps, got 7"):
        module(bad_history)


def test_transformer_rejects_wrong_feature_dim() -> None:
    module = TransformerAdaptationModule(n_steps=8, state_action_dim=6, ext_dim=3)
    bad_history = jnp.ones((8, 5), dtype=jnp.float32)

    with pytest.raises(ValueError, match="expected feature dim 6, got 5"):
        module(bad_history)
