import jax
import jax.numpy as jnp


def estimate_thruster_effectiveness(
    commanded_ctrl: jnp.ndarray,
    applied_ctrl: jnp.ndarray,
    *,
    eps: float = 1e-3,
    max_effectiveness: float = 2.0,
) -> jnp.ndarray:
    """
    Estimate per-thruster control effectiveness from commanded and applied thrust.

    Near-zero commands are uninformative, so those entries default to nominal
    effectiveness. Values above one capture stuck-on or overactive behavior.
    """
    commanded_ctrl = jnp.asarray(commanded_ctrl)
    applied_ctrl = jnp.asarray(applied_ctrl, dtype=commanded_ctrl.dtype)
    informative = jnp.abs(commanded_ctrl) > eps
    ratio = applied_ctrl / jnp.where(informative, commanded_ctrl, 1.0)
    eta = jnp.where(informative, ratio, 1.0)
    return jnp.clip(eta, 0.0, max_effectiveness)


def controllability_metrics(
    effectiveness: jnp.ndarray,
    thruster_mixer_T: jnp.ndarray,
    *,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Compute compact, rollout-safe wrench-authority proxies.

    This function is called inside the compiled rollout scan, so it intentionally
    avoids per-step SVD/eigendecompositions. It returns axis-aligned authority
    proxies `[min_axis_authority, mean_log_axis_authority]` from the diagonal of
    `A A^T`, where `A` maps thruster forces to body wrench.

    Exact SVD/pseudoinverse diagnostics are kept in `authority_metrics_from_wrench`
    and should be used at low logging/evaluation frequency.
    """
    effectiveness = jnp.asarray(effectiveness)
    mixer_T = jnp.asarray(thruster_mixer_T, dtype=effectiveness.dtype)
    effective_mixer_T = effectiveness[:, :, None] * mixer_T[None, :, :]
    wrench_map = jnp.swapaxes(effective_mixer_T, 1, 2)
    axis_authority_sq = jnp.sum(wrench_map * wrench_map, axis=2)
    axis_authority = jnp.sqrt(axis_authority_sq + eps)
    min_axis_authority = jnp.min(axis_authority, axis=1)
    mean_log_axis_authority = jnp.mean(jnp.log(axis_authority + eps), axis=1)
    return jnp.stack([min_axis_authority, mean_log_axis_authority], axis=1)


def authority_metrics_from_wrench(
    *,
    commanded_ctrl: jnp.ndarray,
    applied_ctrl: jnp.ndarray,
    desired_wrench: jnp.ndarray,
    thruster_mixer_T: jnp.ndarray,
    eps: float = 1e-6,
) -> dict[str, jnp.ndarray]:
    """
    Compute task-conditioned authority/recoverability diagnostics.

    `feasibility_error` measures how far the desired wrench is from the
    instantaneous wrench subspace spanned by the effectiveness-scaled mixer.
    This is a linearized diagnostic, not a full bounded-thrust reachability test.
    """
    effectiveness = estimate_thruster_effectiveness(commanded_ctrl, applied_ctrl)
    mixer_T = jnp.asarray(thruster_mixer_T, dtype=effectiveness.dtype)
    effective_mixer_T = effectiveness[:, :, None] * mixer_T[None, :, :]
    wrench_map = jnp.swapaxes(effective_mixer_T, 1, 2)  # [B, 6, n_thrusters]

    singular_values = jax.vmap(
        lambda mat: jnp.linalg.svd(mat, compute_uv=False),
    )(wrench_map)
    min_sv = singular_values[:, -1]
    mean_log_sv = jnp.mean(jnp.log(singular_values + eps), axis=1)
    condition_number = singular_values[:, 0] / (singular_values[:, -1] + eps)

    def _project(mat, wrench):
        return mat @ (jnp.linalg.pinv(mat) @ wrench)

    desired_wrench = jnp.asarray(desired_wrench, dtype=effectiveness.dtype)
    projected = jax.vmap(_project)(wrench_map, desired_wrench)
    feasibility_error = jnp.linalg.norm(desired_wrench - projected, axis=1)
    desired_norm = jnp.linalg.norm(desired_wrench, axis=1)
    normalized_feasibility_error = feasibility_error / (desired_norm + eps)

    return {
        "min_singular_value": min_sv,
        "mean_log_singular_value": mean_log_sv,
        "condition_number": condition_number,
        "wrench_feasibility_error": feasibility_error,
        "normalized_wrench_feasibility_error": normalized_feasibility_error,
    }


def task_authority_targets(
    *,
    commanded_ctrl: jnp.ndarray,
    applied_ctrl: jnp.ndarray,
    actual_wrench: jnp.ndarray,
    desired_wrench: jnp.ndarray,
    eps: float = 1e-6,
) -> jnp.ndarray:
    """
    Lightweight task-conditioned actuator-authority target for AM training.

    This is not a full bounded wrench-polytope solve. It asks a narrower,
    rollout-cheap question: for the current desired wrench, how badly did the
    realized actuator response miss, and how much actuator mismatch was present?
    """
    residual = jnp.asarray(actual_wrench) - jnp.asarray(desired_wrench)
    desired_norm = jnp.linalg.norm(desired_wrench, axis=1)
    residual_norm = jnp.linalg.norm(residual, axis=1)
    normalized_wrench_error = residual_norm / (desired_norm + eps)
    directional_error = jnp.sum(residual * desired_wrench, axis=1) / (
        desired_norm**2 + eps
    )
    ctrl_scale = jnp.abs(commanded_ctrl) + 1e-3
    actuator_mismatch = jnp.mean(
        jnp.clip(jnp.abs(applied_ctrl - commanded_ctrl) / ctrl_scale, 0.0, 10.0),
        axis=1,
    )
    return jnp.stack(
        [normalized_wrench_error, directional_error, actuator_mismatch],
        axis=1,
    )


def summarize_authority_metrics(
    metrics: dict[str, jnp.ndarray],
    mask: jnp.ndarray | None = None,
) -> dict[str, float]:
    """
    Convert per-sample authority metrics into scalar logging values.
    """
    if not metrics:
        return {}
    first_value = next(iter(metrics.values()))
    if mask is None:
        mask_f = jnp.ones_like(first_value, dtype=first_value.dtype)
    else:
        mask_f = jnp.asarray(mask, dtype=first_value.dtype)
    denom = jnp.maximum(mask_f.sum(), 1.0)
    return {
        f"authority/{name}_mean": float((value * mask_f).sum() / denom)
        for name, value in metrics.items()
    }


def authority_bin_stats(
    normalized_feasibility_error: jnp.ndarray,
    success: jnp.ndarray,
    *,
    thresholds: tuple[float, float, float] = (0.05, 0.2, 0.5),
) -> dict[str, float]:
    """
    Report recoverability-bin occupancy and success rates.

    Bins are based on normalized task-conditioned wrench feasibility error:
    easy <= t0, medium <= t1, hard <= t2, extreme > t2.
    """
    values = jnp.asarray(normalized_feasibility_error)
    success_f = jnp.asarray(success, dtype=jnp.float32)
    t0, t1, t2 = thresholds
    bins = {
        "easy": values <= t0,
        "medium": jnp.logical_and(values > t0, values <= t1),
        "hard": jnp.logical_and(values > t1, values <= t2),
        "extreme": values > t2,
    }
    total = jnp.maximum(values.size, 1)
    stats: dict[str, float] = {}
    for name, mask in bins.items():
        mask_f = mask.astype(jnp.float32)
        count = mask_f.sum()
        denom = jnp.maximum(count, 1.0)
        stats[f"authority/bin_{name}_fraction"] = float(count / total)
        stats[f"authority/bin_{name}_success_rate"] = float(
            (success_f * mask_f).sum() / denom
        )
    return stats


def build_adaptive_context(
    *,
    commanded_ctrl: jnp.ndarray,
    applied_ctrl: jnp.ndarray,
    actual_wrench: jnp.ndarray,
    desired_wrench: jnp.ndarray,
    previous_context: jnp.ndarray,
    use_adaptive_approach: bool,
    adaptive_context_mode: str = "structured",
    thruster_mixer_T: jnp.ndarray | None = None,
) -> jnp.ndarray:
    """
    Build the adaptive policy context.

    The first six dimensions remain the wrench residual for RMA compatibility.
    Extra dimensions encode per-thruster effectiveness and wrench authority.
    """
    if not use_adaptive_approach:
        return previous_context

    residual_wrench = actual_wrench - desired_wrench
    if adaptive_context_mode not in (
        "residual",
        "residual_effectiveness",
        "residual_controllability",
        "structured",
    ):
        raise ValueError(f"Unknown adaptive_context_mode: {adaptive_context_mode}")

    structured_context = residual_wrench
    if adaptive_context_mode != "residual" and thruster_mixer_T is not None:
        effectiveness = estimate_thruster_effectiveness(commanded_ctrl, applied_ctrl)
        if adaptive_context_mode == "residual_effectiveness":
            structured_context = jnp.concatenate(
                [residual_wrench, effectiveness], axis=1
            )
        elif adaptive_context_mode == "residual_controllability":
            metrics = controllability_metrics(effectiveness, thruster_mixer_T)
            structured_context = jnp.concatenate([residual_wrench, metrics], axis=1)
        elif adaptive_context_mode == "structured":
            metrics = controllability_metrics(effectiveness, thruster_mixer_T)
            structured_context = jnp.concatenate(
                [residual_wrench, effectiveness, metrics],
                axis=1,
            )

    target_dim = previous_context.shape[-1]
    if structured_context.shape[-1] >= target_dim:
        return structured_context[:, :target_dim]

    pad = jnp.zeros(
        (structured_context.shape[0], target_dim - structured_context.shape[-1]),
        dtype=structured_context.dtype,
    )
    return jnp.concatenate([structured_context, pad], axis=1)


def build_adaptation_query(
    *,
    states: jnp.ndarray,
    desired_wrench: jnp.ndarray,
    use_task_conditioned_am: bool,
) -> jnp.ndarray:
    """
    Build the state/task query used by task-conditioned adaptation modules.

    The query is intentionally compact and available during rollout logging:
    normalized state features plus the desired wrench associated with the
    recent command.
    """
    if not use_task_conditioned_am:
        return jnp.zeros((states.shape[0], 0), dtype=states.dtype)
    return jnp.concatenate([states, desired_wrench], axis=1)
