import inspect
from typing import Any, Callable, Optional

import jax
import jax.numpy as jnp

from .types import (
    FunctionalRolloutCallbacks,
    FunctionalRolloutResult,
    _FunctionalRolloutStep,
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
    reset_fn_accepts_mask = len(inspect.signature(reset_fn).parameters) >= 3
    step_fn_accepts_prev_states = (
        "prev_states" in inspect.signature(step_fn).parameters
    )

    def _initial_episode_state():
        return (
            initial_state,
            initial_residuals,
            rng,
            extra,
            jnp.zeros((num_envs,), dtype=jnp.float32),
            jnp.zeros((num_envs,), dtype=jnp.int32),
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
        reset_pending: jnp.ndarray,
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

        if step_fn_accepts_prev_states:
            next_env_state, step_output = step_fn(
                env_state,
                actions,
                reference_waypoint,
                step_config,
                residuals,
                prev_states=states_curr,
            )
        else:
            next_env_state, step_output = step_fn(
                env_state,
                actions,
                reference_waypoint,
                step_config,
                residuals,
            )

        ep_ret_next = ep_ret + step_output.rewards
        ep_len_next = ep_len + 1

        terminated_mask = step_output.terminals.astype(bool)
        timeout_mask = ep_len_next >= step_config.max_episode_len
        truncated_mask = jnp.logical_and(timeout_mask, jnp.logical_not(terminated_mask))
        done_mask = jnp.logical_or(terminated_mask, truncated_mask)
        epoch_last = jnp.equal(step_idx, num_steps - 1)
        done_flag = jnp.any(done_mask)
        reset_mask = jnp.logical_and(done_mask, jnp.logical_not(epoch_last))

        next_residuals, step_aux, carry_extra = _post_step(
            step_idx, step_output, actions, residuals, reset_mask, carry_extra
        )

        bootstrap_mask = jnp.logical_or(
            truncated_mask,
            jnp.logical_and(epoch_last, jnp.logical_not(terminated_mask)),
        )
        bootstrap_values_raw, rng_key, carry_extra = jax.lax.cond(
            jnp.any(bootstrap_mask),
            lambda args: _bootstrap_value(step_idx, *args),
            lambda args: (jnp.zeros_like(step_output.rewards), args[2], args[3]),
            operand=(next_env_state, next_residuals, rng_key, carry_extra),
        )
        bootstrap_values = jnp.where(
            bootstrap_mask,
            bootstrap_values_raw,
            jnp.zeros_like(bootstrap_values_raw),
        )

        episode_return = jnp.where(
            done_mask,
            ep_ret_next,
            jnp.zeros_like(ep_ret_next),
        )

        def _reset_after_done(mask):
            if reset_fn_accepts_mask:
                reset_state = reset_fn(next_env_state, step_config, mask)
            else:
                reset_state = reset_fn(next_env_state, step_config)
            residual_mask = mask[:, None]
            reset_residuals = jnp.where(
                residual_mask,
                jnp.zeros_like(next_residuals),
                next_residuals,
            )
            reset_returns = jnp.where(mask, jnp.zeros_like(ep_ret_next), ep_ret_next)
            reset_lengths = jnp.where(mask, jnp.zeros_like(ep_len_next), ep_len_next)
            return reset_state, reset_residuals, reset_returns, reset_lengths

        next_env_state, next_residuals, ep_ret_final, ep_len_final = jax.lax.cond(
            jnp.any(reset_mask),
            _reset_after_done,
            lambda _: (next_env_state, next_residuals, ep_ret_next, ep_len_next),
            operand=reset_mask,
        )

        step_record = _FunctionalRolloutStep(
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
    done_masks = steps.done_mask
    terminated_masks = steps.terminated_mask
    truncated_masks = steps.truncated_mask
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
        done_masks=done_masks,
        terminated_masks=terminated_masks,
        truncated_masks=truncated_masks,
        bootstrap_values=bootstrap_values,
        aux=aux,
        final_state=final_state,
        final_residuals=final_residuals,
        final_rng=final_rng,
        final_extra=final_extra,
    )
