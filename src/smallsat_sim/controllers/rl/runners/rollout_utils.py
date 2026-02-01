from dataclasses import dataclass
from typing import Any, Callable, Optional

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
        [int, Any, jnp.ndarray, jnp.ndarray, bool, Any],
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
        bootstrap_value=bootstrap_value,
        aux=aux,
    )


jax.tree_util.register_pytree_node(
    _FunctionalRolloutStep,
    _functional_rollout_step_flatten,
    _functional_rollout_step_unflatten,
)


def run_functional_rollout(
    *,
    step_config: Any,
    initial_state: Any,
    initial_residuals: jnp.ndarray,
    rng: jnp.ndarray,
    num_steps: int,
    reference_waypoint: jnp.ndarray,
    callbacks: FunctionalRolloutCallbacks,
    extra: Any = None,
    state_features_fn: Optional[Callable[[Any, jnp.ndarray], jnp.ndarray]] = None,
    step_fn: Optional[
        Callable[[Any, jnp.ndarray, jnp.ndarray, Any, jnp.ndarray], tuple[Any, Any]]
    ] = None,
    reset_fn: Optional[Callable[[Any, Any], Any]] = None,
) -> FunctionalRolloutResult:
    """
    Advance the vectorised environment purely functionally for ``num_steps``.

    Invariants:
    - The helper mirrors the imperative `VecEnv.transition` semantics, including
      the `control_decimation` inner loop and per-epoch resets when all
      environments terminate or the configured horizon is reached.
    - `initial_residuals` must have shape `(num_envs, res_dim)` (or `(num_envs, 0)`
      when residuals are disabled).
    - Callback implementations must be side-effect free; any mutable state should
      be threaded via the `extra` carry.
    - The returned `step_outputs` contain the per-step state snapshots needed to
      reconstruct rewards, metrics, and logging payloads without reaching back
      into the imperative environment.
    """

    if state_features_fn is None or step_fn is None or reset_fn is None:
        from smallsat_sim.envs import vec_env as vec_env_mod

        if state_features_fn is None:
            state_features_fn = vec_env_mod._compute_state_features
        if step_fn is None:
            step_fn = vec_env_mod.vecenv_step
        if reset_fn is None:
            reset_fn = vec_env_mod.vecenv_reset_to_config

    assert state_features_fn is not None
    assert step_fn is not None
    assert reset_fn is not None

    num_envs = initial_state.mjx_batch.qpos.shape[0]

    def _initial_episode_state():
        return (
            initial_state,
            initial_residuals,
            rng,
            extra,
            jnp.zeros((num_envs,), dtype=jnp.float32),
            jnp.array(0, dtype=jnp.int32),
        )

    def _prepare_policy_input(
        step_idx: int, states: jnp.ndarray, residuals: jnp.ndarray, carry_extra: Any
    ):
        return callbacks.prepare_policy_input(step_idx, states, residuals, carry_extra)

    def _sample_policy(
        step_idx: int,
        policy_input: jnp.ndarray,
        rng_key: jnp.ndarray,
        carry_extra: Any,
    ):
        return callbacks.sample_policy(step_idx, policy_input, rng_key, carry_extra)

    def _post_step(
        step_idx: int,
        step_output: Any,
        actions: jnp.ndarray,
        residuals: jnp.ndarray,
        reset_pending: bool,
        carry_extra: Any,
    ):
        return callbacks.post_step(
            step_idx, step_output, actions, residuals, reset_pending, carry_extra
        )

    def _bootstrap_value(
        step_idx: int,
        env_state: Any,
        residuals: jnp.ndarray,
        rng_key: jnp.ndarray,
        carry_extra: Any,
    ):
        return callbacks.bootstrap_value(
            step_idx, env_state, residuals, rng_key, carry_extra
        )

    def _scan_body(carry, step_idx: int):
        env_state, residuals, rng_key, carry_extra, ep_ret, ep_len = carry

        states_curr = state_features_fn(env_state.mjx_batch, reference_waypoint)
        policy_input, carry_extra = _prepare_policy_input(
            step_idx, states_curr, residuals, carry_extra
        )
        actions, values, logp, rng_key, carry_extra = _sample_policy(
            step_idx, policy_input, rng_key, carry_extra
        )

        next_env_state, step_output = step_fn(
            env_state,
            actions,
            reference_waypoint,
            step_config,
            residuals,
        )

        def _log_nan(_):
            jax.debug.print("NaN in next_obs at step {s}", s=step_idx)
            return jnp.array(0, dtype=jnp.int32)

        _ = jax.lax.cond(
            jnp.isnan(step_output.next_obs).any(),
            _log_nan,
            lambda _: jnp.array(0, dtype=jnp.int32),
            operand=None,
        )

        ep_ret_next = ep_ret + step_output.rewards
        ep_len_next = ep_len + 1

        all_terminal = jnp.all(step_output.terminals)
        timeout = ep_len_next >= step_config.max_episode_len
        epoch_last = jnp.equal(step_idx, num_steps - 1)
        done_without_epoch = jnp.logical_or(all_terminal, timeout)
        done_flag = done_without_epoch

        next_residuals, step_aux, carry_extra = _post_step(
            step_idx, step_output, actions, residuals, done_without_epoch, carry_extra
        )

        bootstrap_condition = jnp.logical_and(
            jnp.logical_or(timeout, epoch_last),
            jnp.logical_not(all_terminal),
        )
        bootstrap_values, rng_key, carry_extra = jax.lax.cond(
            bootstrap_condition,
            lambda args: _bootstrap_value(step_idx, *args),
            lambda args: (jnp.zeros_like(step_output.rewards), args[2], args[3]),
            operand=(next_env_state, next_residuals, rng_key, carry_extra),
        )

        episode_return = jax.lax.cond(
            done_flag,
            lambda _: ep_ret_next,
            lambda _: jnp.zeros_like(ep_ret_next),
            operand=None,
        )

        def _reset_after_done(_):
            reset_state = reset_fn(next_env_state, step_config)
            zero_residuals = jnp.zeros_like(initial_residuals)
            zero_return = jnp.zeros((num_envs,), dtype=ep_ret_next.dtype)
            zero_length = jnp.array(0, dtype=ep_len_next.dtype)
            return reset_state, zero_residuals, zero_return, zero_length

        next_env_state, next_residuals, ep_ret_final, ep_len_final = jax.lax.cond(
            jnp.logical_and(done_without_epoch, jnp.logical_not(epoch_last)),
            _reset_after_done,
            lambda _: (next_env_state, next_residuals, ep_ret_next, ep_len_next),
            operand=None,
        )

        step_record = _FunctionalRolloutStep(
            step_output=step_output,
            actions=actions,
            values=values,
            logp=logp,
            residuals=next_residuals,
            episode_return=episode_return,
            done_flag=done_flag,
            bootstrap_value=bootstrap_values,
            aux=step_aux,
        )

        new_carry = (
            next_env_state,
            next_residuals,
            rng_key,
            carry_extra,
            ep_ret_final,
            ep_len_final,
        )
        return new_carry, step_record

    initial_carry = _initial_episode_state()
    (final_state, final_residuals, final_rng, final_extra, _, _), steps = jax.lax.scan(
        _scan_body,
        initial_carry,
        jnp.arange(num_steps, dtype=jnp.int32),
    )

    step_outputs = steps.step_output
    actions = steps.actions
    values = steps.values
    logp = steps.logp
    residuals = steps.residuals
    episode_returns = steps.episode_return
    done_flags = steps.done_flag
    bootstrap_values = steps.bootstrap_value
    aux = steps.aux

    return FunctionalRolloutResult(
        step_outputs=step_outputs,
        actions=actions,
        values=values,
        logp=logp,
        residuals=residuals,
        episode_returns=episode_returns,
        done_flags=done_flags,
        bootstrap_values=bootstrap_values,
        aux=aux,
        final_state=final_state,
        final_residuals=final_residuals,
        final_rng=final_rng,
        final_extra=final_extra,
    )
