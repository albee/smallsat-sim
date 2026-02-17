from dataclasses import dataclass
from typing import Any, Callable

import jax
import jax.numpy as jnp


@dataclass
class FunctionalRolloutCallbacks:
    """
    Collection of callables used by `run_functional_rollout` to interact with
    policy logic outside the environment stepping loop.

    Each callable receives the current step index together with the evolving
    carry so callers can maintain additional per-rollout state (e.g. history
    buffers for the adaptation module).
    """

    prepare_policy_input: Callable[
        [int, jnp.ndarray, jnp.ndarray, Any], tuple[jnp.ndarray, Any]
    ]
    sample_policy: Callable[
        [int, jnp.ndarray, jnp.ndarray, Any],
        tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any],
    ]
    post_step: Callable[
        [int, Any, jnp.ndarray, jnp.ndarray, jnp.ndarray, Any],
        tuple[jnp.ndarray, Any, Any],
    ]
    bootstrap_value: Callable[
        [int, Any, jnp.ndarray, jnp.ndarray, Any],
        tuple[jnp.ndarray, jnp.ndarray, Any],
    ]


@dataclass
class FunctionalRolloutResult:
    """
    Batched outputs produced by `run_functional_rollout`.
    """

    step_outputs: Any
    actions: jnp.ndarray
    values: jnp.ndarray
    logp: jnp.ndarray
    residuals: jnp.ndarray
    episode_returns: jnp.ndarray
    done_flags: jnp.ndarray
    done_masks: jnp.ndarray
    terminated_masks: jnp.ndarray
    truncated_masks: jnp.ndarray
    bootstrap_values: jnp.ndarray
    aux: Any
    final_state: Any
    final_residuals: jnp.ndarray
    final_rng: jnp.ndarray
    final_extra: Any


@dataclass
class AdaptationRolloutExtra:
    history: jnp.ndarray
    counts: jnp.ndarray


def _adaptation_rollout_extra_flatten(extra: "AdaptationRolloutExtra"):
    children = (extra.history, extra.counts)
    return children, None


def _adaptation_rollout_extra_unflatten(aux_data, children):
    history, counts = children
    return AdaptationRolloutExtra(history=history, counts=counts)


jax.tree_util.register_pytree_node(
    AdaptationRolloutExtra,
    _adaptation_rollout_extra_flatten,
    _adaptation_rollout_extra_unflatten,
)


@dataclass
class _FunctionalRolloutStep:
    step_output: Any
    actions: jnp.ndarray
    values: jnp.ndarray
    logp: jnp.ndarray
    residuals: jnp.ndarray
    episode_return: jnp.ndarray
    done_flag: jnp.ndarray
    done_mask: jnp.ndarray
    terminated_mask: jnp.ndarray
    truncated_mask: jnp.ndarray
    bootstrap_value: jnp.ndarray
    aux: Any


def _functional_rollout_step_flatten(step: "_FunctionalRolloutStep"):
    children = (
        step.step_output,
        step.actions,
        step.values,
        step.logp,
        step.residuals,
        step.episode_return,
        step.done_flag,
        step.done_mask,
        step.terminated_mask,
        step.truncated_mask,
        step.bootstrap_value,
        step.aux,
    )
    return children, None


def _functional_rollout_step_unflatten(aux_data, children):
    (
        step_output,
        actions,
        values,
        logp,
        residuals,
        episode_return,
        done_flag,
        done_mask,
        terminated_mask,
        truncated_mask,
        bootstrap_value,
        aux,
    ) = children
    return _FunctionalRolloutStep(
        step_output=step_output,
        actions=actions,
        values=values,
        logp=logp,
        residuals=residuals,
        episode_return=episode_return,
        done_flag=done_flag,
        done_mask=done_mask,
        terminated_mask=terminated_mask,
        truncated_mask=truncated_mask,
        bootstrap_value=bootstrap_value,
        aux=aux,
    )


jax.tree_util.register_pytree_node(
    _FunctionalRolloutStep,
    _functional_rollout_step_flatten,
    _functional_rollout_step_unflatten,
)

