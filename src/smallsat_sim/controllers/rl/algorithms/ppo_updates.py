from functools import partial

import jax
import jax.numpy as jnp
from flax import nnx


def actor_loss_fn(
    actor_model,
    tdres: jnp.ndarray,
    obs_residuals: jnp.ndarray,
    actions: jnp.ndarray,
    logp: jnp.ndarray,
    clip_ratio: float,
    entropy_coef: float,
):
    pi, logp_a = actor_model.forward(obs_residuals, actions)
    ratio = jnp.exp(logp_a - logp)
    clipped_ratio = jax.lax.clamp(1 - clip_ratio, ratio, 1 + clip_ratio)
    surrogate = jax.lax.min(ratio * tdres, clipped_ratio * tdres)
    entropy_scale = jnp.asarray(entropy_coef, dtype=tdres.dtype)
    entropy_bonus = entropy_scale * pi.entropy()
    return -(surrogate.mean() + entropy_bonus.mean())


# Define the critic loss
def critic_loss_fn(
    critic_model,
    returns: jnp.ndarray,
    obs_residuals: jnp.ndarray,
):
    values = critic_model.forward(obs_residuals)
    return jnp.mean((values - returns) ** 2)  # MSE loss


def _diag_gaussian_kl(
    mu_old: jnp.ndarray,
    std_old: jnp.ndarray,
    mu_new: jnp.ndarray,
    std_new: jnp.ndarray,
) -> jnp.ndarray:
    """
    Compute KL(N_old || N_new) for diagonal Gaussians. Returns per-sample KL.
    """
    log_std_ratio = jnp.log(std_new) - jnp.log(std_old)
    numerator = std_old**2 + (mu_old - mu_new) ** 2
    denom = 2.0 * (std_new**2)
    kl_per_dim = log_std_ratio + numerator / denom - 0.5
    return jnp.sum(kl_per_dim, axis=-1)


def _shuffle_and_batch(
    array: jnp.ndarray,
    permutation: jnp.ndarray,
    num_minibatches: int,
    minibatch_size: int,
) -> jnp.ndarray:
    effective_batch_size = num_minibatches * minibatch_size
    shuffled = array[permutation]
    trimmed = shuffled[:effective_batch_size]
    return trimmed.reshape((num_minibatches, minibatch_size) + trimmed.shape[1:])


@partial(jax.jit, static_argnums=(12, 13))
def _actor_epochs_jit(
    graphdef,
    state,
    actor_graphdef,
    old_actor_state,
    epoch_keys: jnp.ndarray,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    logp: jnp.ndarray,
    clip_ratio: float,
    entropy_coef: float,
    target_kl: float,
    num_minibatches: int,
    minibatch_size: int,
):
    batch_size = obs.shape[0]
    zero_loss = jnp.zeros((), dtype=advantages.dtype)
    zero_kl = jnp.zeros((), dtype=logp.dtype)
    zero_clip = jnp.zeros((), dtype=logp.dtype)
    zero_steps = jnp.zeros((), dtype=jnp.int32)

    def epoch_body(i, carry):
        (
            state,
            stop,
            last_loss,
            last_kl,
            kl_sum,
            clip_sum,
            step_count,
            last_clip,
        ) = carry
        key = epoch_keys[i]

        def run_epoch(args):
            (
                state,
                stop,
                last_loss,
                last_kl,
                kl_sum,
                clip_sum,
                step_count,
                last_clip,
                epoch_key,
            ) = args
            permutation = jax.random.permutation(epoch_key, batch_size)
            obs_batches = _shuffle_and_batch(
                obs, permutation, num_minibatches, minibatch_size
            )
            act_batches = _shuffle_and_batch(
                actions, permutation, num_minibatches, minibatch_size
            )
            adv_batches = _shuffle_and_batch(
                advantages, permutation, num_minibatches, minibatch_size
            )
            logp_batches = _shuffle_and_batch(
                logp, permutation, num_minibatches, minibatch_size
            )

            def minibatch_body(mb, carry):
                (
                    state_mb,
                    stop_mb,
                    last_loss_mb,
                    last_kl_mb,
                    kl_sum_mb,
                    clip_sum_mb,
                    step_count_mb,
                    last_clip_mb,
                ) = carry
                operand = (
                    state_mb,
                    stop_mb,
                    last_loss_mb,
                    last_kl_mb,
                    kl_sum_mb,
                    clip_sum_mb,
                    step_count_mb,
                    last_clip_mb,
                    obs_batches[mb],
                    act_batches[mb],
                    adv_batches[mb],
                    logp_batches[mb],
                )

                def skip_fn(op):
                    state_skip, stop_skip, loss_skip, kl_skip, kl_sum_skip, clip_sum_skip, step_skip, clip_skip, *_ = op
                    return (
                        state_skip,
                        stop_skip,
                        loss_skip,
                        kl_skip,
                        kl_sum_skip,
                        clip_sum_skip,
                        step_skip,
                        clip_skip,
                    )

                def update_fn(op):
                    (
                        state_upd,
                        stop_upd,
                        _,
                        _,
                        kl_sum_upd,
                        clip_sum_upd,
                        step_count_upd,
                        _,
                        obs_mb,
                        act_mb,
                        adv_mb,
                        logp_mb,
                    ) = op
                    actor, optimizer = nnx.merge(graphdef, state_upd)

                    def loss_fn(model):
                        return actor_loss_fn(
                            model,
                            adv_mb,
                            obs_mb,
                            act_mb,
                            logp_mb,
                            clip_ratio,
                            entropy_coef,
                        )

                    loss, grads = nnx.value_and_grad(loss_fn)(actor)
                    optimizer.update(grads)
                    pi_new = actor._distribution(obs_mb)
                    _, logp_a = actor.forward(obs_mb, act_mb)
                    ratio = jnp.exp(logp_a - logp_mb)
                    clip_fraction = jnp.mean(
                        jnp.logical_or(
                            ratio > (1 + clip_ratio), ratio < (1 - clip_ratio)
                        )
                    )
                    actor_old = nnx.merge(actor_graphdef, old_actor_state)
                    pi_old = actor_old._distribution(obs_mb)
                    kl = jnp.mean(
                        _diag_gaussian_kl(
                            pi_old.mean(),
                            pi_old.stddev(),
                            pi_new.mean(),
                            pi_new.stddev(),
                        )
                    )
                    new_stop = jnp.logical_or(stop_upd, kl > 1.5 * target_kl)
                    new_state = nnx.state((actor, optimizer))
                    return (
                        new_state,
                        new_stop,
                        loss,
                        kl,
                        kl_sum_upd + kl,
                        clip_sum_upd + clip_fraction,
                        step_count_upd + 1,
                        clip_fraction,
                    )

                return jax.lax.cond(stop_mb, skip_fn, update_fn, operand)

            return jax.lax.fori_loop(
                0,
                num_minibatches,
                minibatch_body,
                (
                    state,
                    stop,
                    last_loss,
                    last_kl,
                    kl_sum,
                    clip_sum,
                    step_count,
                    last_clip,
                ),
            )

        return jax.lax.cond(
            stop,
            lambda op: op[:8],
            run_epoch,
            (
                state,
                stop,
                last_loss,
                last_kl,
                kl_sum,
                clip_sum,
                step_count,
                last_clip,
                key,
            ),
        )

    (
        state,
        stop,
        last_loss,
        last_kl,
        kl_sum,
        clip_sum,
        step_count,
        last_clip,
    ) = jax.lax.fori_loop(
        0,
        epoch_keys.shape[0],
        epoch_body,
        (state, False, zero_loss, zero_kl, zero_kl, zero_clip, zero_steps, zero_clip),
    )

    mean_kl = jnp.where(step_count > 0, kl_sum / step_count, 0.0)
    mean_clip = jnp.where(step_count > 0, clip_sum / step_count, 0.0)

    return last_loss, last_kl, mean_kl, mean_clip, state


@partial(jax.jit, static_argnums=(6, 7, 8, 9, 10))
def _critic_epochs_jit(
    graphdef,
    state,
    epoch_keys: jnp.ndarray,
    obs: jnp.ndarray,
    returns: jnp.ndarray,
    old_values: jnp.ndarray,
    use_value_clip: bool,
    value_clip_coef: float,
    debug_prints: bool,
    num_minibatches: int,
    minibatch_size: int,
):
    batch_size = obs.shape[0]
    zero_loss = jnp.zeros((), dtype=returns.dtype)

    def epoch_body(i, carry):
        state, last_loss = carry
        key = epoch_keys[i]
        permutation = jax.random.permutation(key, batch_size)
        obs_batches = _shuffle_and_batch(
            obs, permutation, num_minibatches, minibatch_size
        )
        returns_batches = _shuffle_and_batch(
            returns, permutation, num_minibatches, minibatch_size
        )
        old_values_batches = _shuffle_and_batch(
            old_values, permutation, num_minibatches, minibatch_size
        )

        def minibatch_body(mb, carry):
            state_mb, _ = carry
            critic, optimizer = nnx.merge(graphdef, state_mb)

            def loss_fn(model):
                vals = model.forward(obs_batches[mb])
                if use_value_clip:
                    v_old = old_values_batches[mb]
                    v_clipped = v_old + jnp.clip(
                        vals - v_old, -value_clip_coef, value_clip_coef
                    )
                    loss_unclipped = (vals - returns_batches[mb]) ** 2
                    loss_clipped = (v_clipped - returns_batches[mb]) ** 2
                    # Debug prints: report minibatch-level summaries so we can
                    # verify that clipping is happening and inspect magnitudes.
                    if debug_prints:
                        jax.debug.print(
                            "[PPO DEBUG] critic clip coef={} mean_delta={:.6f} mean_loss_unclipped={:.6f} mean_loss_clipped={:.6f} mean_return={:.6f} std_return={:.6f} mean_value={:.6f} std_value={:.6f}",
                            value_clip_coef,
                            jnp.mean(vals - v_old),
                            jnp.mean(loss_unclipped),
                            jnp.mean(loss_clipped),
                            jnp.mean(returns_batches[mb]),
                            jnp.std(returns_batches[mb]),
                            jnp.mean(v_old),
                            jnp.std(v_old),
                        )
                    return jnp.mean(jnp.maximum(loss_unclipped, loss_clipped))
                else:
                    return jnp.mean((vals - returns_batches[mb]) ** 2)

            loss, grads = nnx.value_and_grad(loss_fn)(critic)
            optimizer.update(grads)
            new_state = nnx.state((critic, optimizer))
            return new_state, loss

        return jax.lax.fori_loop(0, num_minibatches, minibatch_body, (state, last_loss))

    state, last_loss = jax.lax.fori_loop(
        0, epoch_keys.shape[0], epoch_body, (state, zero_loss)
    )

    return last_loss, state


@partial(jax.jit, static_argnums=(14, 15, 16, 17, 18))
def _actor_critic_epochs_jit(
    graphdef,
    state,
    actor_graphdef,
    old_actor_state,
    epoch_keys: jnp.ndarray,
    obs: jnp.ndarray,
    actions: jnp.ndarray,
    advantages: jnp.ndarray,
    logp: jnp.ndarray,
    returns: jnp.ndarray,
    old_values: jnp.ndarray,
    clip_ratio: float,
    entropy_coef: float,
    target_kl: float,
    use_value_clip: bool,
    value_clip_coef: float,
    debug_prints: bool,
    num_minibatches: int,
    minibatch_size: int,
    critic_grad_scale: float,
):
    batch_size = obs.shape[0]
    zero_actor_loss = jnp.zeros((), dtype=advantages.dtype)
    zero_critic_loss = jnp.zeros((), dtype=returns.dtype)
    zero_actor_loss_sum = jnp.zeros((), dtype=advantages.dtype)
    zero_critic_loss_sum = jnp.zeros((), dtype=returns.dtype)
    zero_steps = jnp.zeros((), dtype=jnp.int32)
    zero_kl = jnp.zeros((), dtype=logp.dtype)
    zero_clip = jnp.zeros((), dtype=logp.dtype)

    def epoch_body(i, carry):
        (
            state,
            stop,
            last_actor_loss,
            last_critic_loss,
            last_actor_sum,
            last_critic_sum,
            last_steps,
            last_kl,
            last_kl_sum,
            last_clip_sum,
            last_clip,
        ) = carry
        key = epoch_keys[i]

        def run_epoch(args):
            (
                state,
                stop,
                last_actor_loss,
                last_critic_loss,
                last_actor_sum,
                last_critic_sum,
                last_steps,
                last_kl,
                last_kl_sum,
                last_clip_sum,
                last_clip,
                epoch_key,
            ) = args
            permutation = jax.random.permutation(epoch_key, batch_size)
            obs_batches = _shuffle_and_batch(
                obs, permutation, num_minibatches, minibatch_size
            )
            act_batches = _shuffle_and_batch(
                actions, permutation, num_minibatches, minibatch_size
            )
            adv_batches = _shuffle_and_batch(
                advantages, permutation, num_minibatches, minibatch_size
            )
            logp_batches = _shuffle_and_batch(
                logp, permutation, num_minibatches, minibatch_size
            )
            returns_batches = _shuffle_and_batch(
                returns, permutation, num_minibatches, minibatch_size
            )
            old_values_batches = _shuffle_and_batch(
                old_values, permutation, num_minibatches, minibatch_size
            )

            def minibatch_body(mb, carry):
                (
                    state_mb,
                    stop_mb,
                    last_actor_loss_mb,
                    last_critic_loss_mb,
                    last_actor_sum_mb,
                    last_critic_sum_mb,
                    last_steps_mb,
                    last_kl_mb,
                    last_kl_sum_mb,
                    last_clip_sum_mb,
                    last_clip_mb,
                ) = carry
                operand = (
                    state_mb,
                    stop_mb,
                    last_actor_loss_mb,
                    last_critic_loss_mb,
                    last_actor_sum_mb,
                    last_critic_sum_mb,
                    last_steps_mb,
                    last_kl_mb,
                    last_kl_sum_mb,
                    last_clip_sum_mb,
                    last_clip_mb,
                    obs_batches[mb],
                    act_batches[mb],
                    adv_batches[mb],
                    logp_batches[mb],
                    returns_batches[mb],
                    old_values_batches[mb],
                )

                def skip_fn(op):
                    # Preserve the running statistics when we skip an update.
                    return op[:11]

                def update_fn(op):
                    (
                        state_upd,
                        stop_upd,
                        _,
                        _,
                        actor_sum_upd,
                        critic_sum_upd,
                        steps_upd,
                        _,
                        kl_sum_upd,
                        clip_sum_upd,
                        _,
                        obs_mb,
                        act_mb,
                        adv_mb,
                        logp_mb,
                        returns_mb,
                        old_values_mb,
                    ) = op
                    actor, actor_opt, critic, critic_opt = nnx.merge(
                        graphdef, state_upd
                    )

                    def actor_loss_local(model):
                        return actor_loss_fn(
                            model,
                            adv_mb,
                            obs_mb,
                            act_mb,
                            logp_mb,
                            clip_ratio,
                            entropy_coef,
                        )

                    actor_loss, actor_grads = nnx.value_and_grad(actor_loss_local)(
                        actor
                    )
                    actor_opt.update(actor_grads)

                    def critic_loss_local(model):
                        vals = model.forward(obs_mb)
                        if use_value_clip:
                            v_old = old_values_mb
                            v_clipped = v_old + jnp.clip(
                                vals - v_old, -value_clip_coef, value_clip_coef
                            )
                            loss_unclipped = (vals - returns_mb) ** 2
                            loss_clipped = (v_clipped - returns_mb) ** 2
                            # Debug prints for actor-critic joint update minibatch
                            if debug_prints:
                                jax.debug.print(
                                    "[PPO DEBUG] joint critic clip coef={} mean_delta={:.6f} mean_loss_unclipped={:.6f} mean_loss_clipped={:.6f} mean_return={:.6f} std_return={:.6f} mean_value={:.6f} std_value={:.6f}",
                                    value_clip_coef,
                                    jnp.mean(vals - v_old),
                                    jnp.mean(loss_unclipped),
                                    jnp.mean(loss_clipped),
                                    jnp.mean(returns_mb),
                                    jnp.std(returns_mb),
                                    jnp.mean(v_old),
                                    jnp.std(v_old),
                                )
                            return jnp.mean(jnp.maximum(loss_unclipped, loss_clipped))
                        else:
                            return jnp.mean((vals - returns_mb) ** 2)

                    critic_loss, critic_grads = nnx.value_and_grad(
                        critic_loss_local
                    )(critic)
                    critic_grad_scale_f = jnp.asarray(
                        critic_grad_scale, dtype=critic_loss.dtype
                    )
                    critic_grads = jax.tree_util.tree_map(
                        lambda g: g * critic_grad_scale_f, critic_grads
                    )
                    critic_opt.update(critic_grads)

                    pi_new = actor._distribution(obs_mb)
                    _, logp_a = actor.forward(obs_mb, act_mb)
                    ratio = jnp.exp(logp_a - logp_mb)
                    clip_fraction = jnp.mean(
                        jnp.logical_or(
                            ratio > (1 + clip_ratio), ratio < (1 - clip_ratio)
                        )
                    )
                    actor_old = nnx.merge(actor_graphdef, old_actor_state)
                    pi_old = actor_old._distribution(obs_mb)
                    kl = jnp.mean(
                        _diag_gaussian_kl(
                            pi_old.mean(),
                            pi_old.stddev(),
                            pi_new.mean(),
                            pi_new.stddev(),
                        )
                    )
                    new_stop = jnp.logical_or(stop_upd, kl > 1.5 * target_kl)
                    new_state = nnx.state((actor, actor_opt, critic, critic_opt))
                    return (
                        new_state,
                        new_stop,
                        actor_loss,
                        critic_loss,
                        actor_sum_upd + actor_loss,
                        critic_sum_upd + critic_loss,
                        steps_upd + 1,
                        kl,
                        kl_sum_upd + kl,
                        clip_sum_upd + clip_fraction,
                        clip_fraction,
                    )

                return jax.lax.cond(stop_mb, skip_fn, update_fn, operand)
            return jax.lax.fori_loop(
                0,
                num_minibatches,
                minibatch_body,
                (
                    state,
                    stop,
                    last_actor_loss,
                    last_critic_loss,
                    last_actor_sum,
                    last_critic_sum,
                    last_steps,
                    last_kl,
                    last_kl_sum,
                    last_clip_sum,
                    last_clip,
                ),
            )

        return jax.lax.cond(
            stop,
            lambda op: op[:11],
            run_epoch,
            (
                state,
                stop,
                last_actor_loss,
                last_critic_loss,
                last_actor_sum,
                last_critic_sum,
                last_steps,
                last_kl,
                last_kl_sum,
                last_clip_sum,
                last_clip,
                key,
            ),
        )

    (
        state,
        stop,
        last_actor_loss,
        last_critic_loss,
        actor_loss_sum,
        critic_loss_sum,
        step_count,
        last_kl,
        kl_sum,
        clip_sum,
        last_clip,
    ) = jax.lax.fori_loop(
        0,
        epoch_keys.shape[0],
        epoch_body,
        (
            state,
            False,
            zero_actor_loss,
            zero_critic_loss,
            zero_actor_loss_sum,
            zero_critic_loss_sum,
            zero_steps,
            zero_kl,
            zero_kl,
            zero_clip,
            zero_clip,
        ),
    )

    mean_actor_loss = jnp.where(step_count > 0, actor_loss_sum / step_count, 0.0)
    mean_critic_loss = jnp.where(step_count > 0, critic_loss_sum / step_count, 0.0)
    mean_kl = jnp.where(step_count > 0, kl_sum / step_count, 0.0)
    mean_clip = jnp.where(step_count > 0, clip_sum / step_count, 0.0)

    return (
        last_actor_loss,
        last_critic_loss,
        mean_actor_loss,
        mean_critic_loss,
        last_kl,
        mean_kl,
        mean_clip,
        state,
    )
