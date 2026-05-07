from typing import Any, Callable

import jax.numpy as jnp

from .types import AdaptationRolloutExtra


def prepare_policy_input_with_residuals(
    _step: int,
    states: jnp.ndarray,
    residuals: jnp.ndarray,
    carry_extra: Any,
) -> tuple[jnp.ndarray, Any]:
    del _step
    if residuals.shape[-1]:
        return jnp.concatenate([states, residuals], axis=1), carry_extra
    return states, carry_extra


def residuals_from_wrench_delta(
    *,
    step_output: Any,
    residuals: jnp.ndarray,
    use_adaptive_approach: bool,
) -> jnp.ndarray:
    if use_adaptive_approach:
        return step_output.actual_wrench - step_output.desired_wrench
    return residuals


def make_zero_bootstrap_value(
    num_envs: int,
) -> Callable[[int, Any, jnp.ndarray, jnp.ndarray, Any], tuple[jnp.ndarray, jnp.ndarray, Any]]:
    def _bootstrap(_step, env_state, residuals, rng_key, carry_extra):
        del _step, env_state, residuals
        zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
        return zeros, rng_key, carry_extra

    return _bootstrap


def update_history_buffer(
    *,
    carry_extra: AdaptationRolloutExtra,
    prev_states: jnp.ndarray,
    actions: jnp.ndarray,
    reset_flag: jnp.ndarray,
    history_len: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, AdaptationRolloutExtra]:
    history = carry_extra.history
    counts = carry_extra.counts
    combined = jnp.concatenate([prev_states, actions], axis=1)
    history = jnp.roll(history, shift=-1, axis=1)
    history = history.at[:, -1, :].set(combined)
    counts = jnp.minimum(counts + 1, history_len)
    history_full = counts >= history_len
    history = jnp.where(reset_flag[:, None, None], jnp.zeros_like(history), history)
    counts = jnp.where(reset_flag, jnp.zeros_like(counts), counts)
    new_extra = AdaptationRolloutExtra(history=history, counts=counts)
    return history, counts, history_full, new_extra
