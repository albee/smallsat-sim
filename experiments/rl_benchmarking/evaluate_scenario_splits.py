"""Evaluate a trained policy on task-feasibility failure-scenario splits."""

from __future__ import annotations

import argparse
import csv
import os
from argparse import Namespace
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax import nnx
from mujoco import mjx

from smallsat_sim.controllers.rl.runners.failure_scenarios import (
    LABEL_BIAS_DOMINATED,
    LABEL_COUPLED_FORCE_TORQUE,
    LABEL_FORCE_DEGENERATE,
    LABEL_NEAR_DEPENDENT,
    LABEL_NONLINEAR_MISMATCH,
    LABEL_SATURATION_PRONE,
    LABEL_TORQUE_DEGENERATE,
    SPLIT_TRAIN,
    TASK_REGIME_EASY_FEASIBLE,
    TASK_REGIME_HARD_FEASIBLE,
    TASK_REGIME_INFEASIBLE,
    TASK_REGIME_NEAR_INFEASIBLE,
    apply_failure_scenarios,
    apply_sampled_failure_scenario_split,
    build_failure_scenario_table,
    scenario_selection_payload,
    scenario_split_counts,
    scenario_task_regime_counts,
    sample_scenario_indices,
    task_wrench_from_state_features,
    targeted_pose_errors_from_scenarios,
)
from smallsat_sim.controllers.rl.runners.runner_utils import load_trained_modules
from smallsat_sim.controllers.rl.runners.rollout import (
    AdaptationRolloutExtra,
    FunctionalRolloutCallbacks,
    make_zero_bootstrap_value,
    prepare_policy_input_with_residuals,
    run_functional_rollout,
    update_history_buffer,
)
from smallsat_sim.controllers.rl.runners.adaptive_context import (
    build_adaptation_query,
    build_adaptive_context,
)
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.vec_env import (
    _compute_state_features,
    _compute_freeflyer_state_features,
    freeflyer_reset_masked,
    vecenv_step_freeflyer,
)
from smallsat_sim.envs.vec_env_types import FreeFlyerVecEnvState, VecEnvState
from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--headless", action="store_true", default=True)
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--log", action="store_true")
    parser.add_argument("--video", action="store_true")
    parser.add_argument("--num-envs", type=int, default=None)
    parser.add_argument("--episodes", type=int, default=3)
    parser.add_argument("--failure-fraction", type=float, default=0.5)
    parser.add_argument("--targeted-failure-fraction", type=float, default=1.0)
    parser.add_argument("--targeted-start-distance", type=float, default=2.0)
    parser.add_argument("--skip-targeted", action="store_true")
    parser.add_argument(
        "--context-ablation",
        default="real",
        choices=("real", "zero", "shuffled", "all"),
        help="Adaptive context mode at eval. Use 'all' for real/zero/shuffled.",
    )
    parser.add_argument(
        "--include-infeasible",
        action="store_true",
        help="Also evaluate infeasible stress splits. Disabled by default.",
    )
    parser.add_argument("--disturbance-fraction", type=float, default=0.0)
    parser.add_argument(
        "--run-name",
        default=os.environ.get("SCENARIO_EVAL_RUN_NAME", "ppo_plain"),
    )
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument(
        "--phase",
        type=int,
        default=2,
        choices=(1, 2),
        help=(
            "Adaptive evaluation phase. Use phase 2 for deployable AM inputs. "
            "Phase 1 is rejected for adaptive runs because it uses privileged labels."
        ),
    )
    parser.add_argument("--use-adaptive-approach", action="store_true")
    parser.add_argument(
        "--am-architecture",
        default=None,
        choices=(None, "transformer", "transformer_cross_attention", "cnn"),
    )
    parser.add_argument(
        "--adaptive-context-mode",
        default=None,
        choices=(
            None,
            "residual",
            "residual_effectiveness",
            "residual_controllability",
            "structured",
        ),
    )
    parser.add_argument("--task-conditioned", action="store_true")
    parser.add_argument("--predict-delta-weight", type=float, default=None)
    parser.add_argument("--predict-tracking-weight", type=float, default=None)
    parser.add_argument("--predict-authority-weight", type=float, default=None)
    parser.add_argument(
        "--output",
        default=None,
    )
    parser.add_argument(
        "--min-samples-per-split",
        type=int,
        default=64,
        help="Minimum sampled scenarios required for a split/pathology report.",
    )
    parser.add_argument(
        "--enforce-min-samples",
        action="store_true",
        help="If set, fail evaluation when sampled scenario count is below threshold.",
    )
    parser.add_argument(
        "--ci-bootstrap-samples",
        type=int,
        default=500,
        help="Bootstrap replicates for confidence intervals.",
    )
    parser.add_argument(
        "--ci-alpha",
        type=float,
        default=0.05,
        help="Two-sided CI level alpha (e.g., 0.05 => 95% CI).",
    )
    return parser.parse_args()


def _sim_args(args: argparse.Namespace) -> Namespace:
    return Namespace(
        headless=args.headless,
        video=False,
        log=args.log,
        wandb=args.wandb,
        num_bodies=1,
        dock_site="dock_orion_port_a",
        dock_approach_offset=None,
        dock_surface_offset=0.0,
    )


def _load_modules(runner: OnPolicyRunner, checkpoint: str | None, phase: int) -> str:
    ckpt_name = checkpoint or runner.training_state_file_name
    ckpt_path = os.path.join(runner.ckpt_dir, ckpt_name)
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}. Train ppo_plain first or pass --checkpoint."
        )

    restored = load_trained_modules(runner.ckpt_dir, ckpt_name)
    actor_state = restored["actor_model"]
    critic_state = restored["critic_model"]
    if isinstance(actor_state, dict):
        nnx.update(runner.agent.actor, actor_state)
    else:
        nnx.update(runner.agent.actor.mu_net, actor_state.mu_net)
    if isinstance(critic_state, dict):
        nnx.update(runner.agent.critic, critic_state)
    else:
        nnx.update(runner.agent.critic.v_net, critic_state.v_net)
    if runner.env.use_adaptive_approach and phase != 2:
        raise ValueError(
            "Adaptive scenario evaluation must use phase=2. Phase 1 consumes "
            "privileged simulator force/wrench labels."
        )
    if runner.env.use_adaptive_approach and phase == 2:
        am_path = os.path.join(runner.ckpt_dir, runner.adaptation_module_file_name)
        if not os.path.isfile(am_path):
            raise FileNotFoundError(
                f"Adaptation module checkpoint not found: {am_path}. "
                "Pass flags matching the trained adaptive method."
            )
        am_state = load_trained_modules(
            runner.ckpt_dir, runner.adaptation_module_file_name
        )["am_model"]
        nnx.update(runner.am, am_state)
    return ckpt_name


def _quat_from_neg_log_error(error: jnp.ndarray) -> jnp.ndarray:
    rotvec = -error
    angle = jnp.linalg.norm(rotvec, axis=1, keepdims=True)
    axis = rotvec / (angle + 1e-6)
    half = 0.5 * angle
    quat_vec = axis * jnp.sin(half)
    quat = jnp.concatenate([jnp.cos(half), quat_vec], axis=1)
    return quat / (jnp.linalg.norm(quat, axis=1, keepdims=True) + 1e-6)


def _with_targeted_initial_offsets(
    initial_state,
    step_config,
    reference_waypoint: jnp.ndarray,
    offsets: jnp.ndarray | None,
    attitude_errors: jnp.ndarray | None = None,
):
    if offsets is None:
        return initial_state

    reference = jnp.asarray(reference_waypoint)
    ref_pos = reference[0, :3] if reference.ndim == 2 else reference[:3]
    num_envs = offsets.shape[0]
    qpos_template = step_config.init_qpos[0]
    qvel_template = step_config.init_qvel[0]
    qpos = jnp.broadcast_to(qpos_template, (num_envs, qpos_template.shape[0]))
    qvel = jnp.broadcast_to(qvel_template, (num_envs, qvel_template.shape[0]))
    qpos = qpos.at[:, :3].set(ref_pos[None, :] + offsets)
    if attitude_errors is None:
        quat = jnp.tile(
            jnp.array([[1.0, 0.0, 0.0, 0.0]], dtype=qpos.dtype),
            (num_envs, 1),
        )
    else:
        quat = _quat_from_neg_log_error(attitude_errors).astype(qpos.dtype)
    qpos = qpos.at[:, 3:7].set(quat)
    qvel = jnp.zeros_like(qvel)

    if isinstance(initial_state, FreeFlyerVecEnvState):
        return initial_state.replace(
            qpos=qpos,
            vel_body=qvel[:, :3],
            omega=qvel[:, 3:6],
            time=jnp.zeros((num_envs,), dtype=qpos.dtype),
            ctrl=jnp.zeros_like(initial_state.ctrl),
            actuator_force=jnp.zeros_like(initial_state.actuator_force),
            terminal_hold_counts=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    if isinstance(initial_state, VecEnvState):
        mjx_batch = initial_state.mjx_batch.replace(qpos=qpos, qvel=qvel)
        mjx_batch = jax.vmap(mjx.forward, in_axes=(None, 0))(
            step_config.mjx_model, mjx_batch
        )
        return initial_state.replace(
            mjx_batch=mjx_batch,
            terminal_hold_counts=jnp.zeros((num_envs,), dtype=jnp.int32),
        )

    raise TypeError(f"Unsupported rollout state type: {type(initial_state)!r}")


def _sample_start_time(key: jnp.ndarray, low: float, high: float) -> float:
    if high <= low:
        return low
    return float(jax.random.uniform(key, (), minval=low, maxval=high))


def _scenario_filter_has_rows(
    table: dict[str, jnp.ndarray],
    *,
    split_id: int,
    task_feasibility_regime: int | None = None,
    difficulty_bin: int | None = None,
    failure_type: int | None = None,
    authority_label_any_mask: int | None = None,
) -> bool:
    mask = table["split"] == int(split_id)
    if task_feasibility_regime is not None:
        mask = jnp.logical_and(
            mask, table["task_feasibility_regime"] == int(task_feasibility_regime)
        )
    if difficulty_bin is not None:
        mask = jnp.logical_and(mask, table["difficulty_bin"] == int(difficulty_bin))
    if failure_type is not None:
        mask = jnp.logical_and(
            mask, jnp.any(table["failure_types"] == int(failure_type), axis=1)
        )
    if authority_label_any_mask is not None:
        mask = jnp.logical_and(
            mask,
            (table["authority_label_mask"] & int(authority_label_any_mask)) != 0,
        )
    return bool(jax.device_get(jnp.any(mask)))


def _rollout_once(
    runner: OnPolicyRunner,
    *,
    initial_position_offsets: jnp.ndarray | None = None,
    initial_attitude_errors: jnp.ndarray | None = None,
    phase: int = 1,
    context_ablation: str = "real",
) -> dict[str, float]:
    num_envs = runner.env.num_envs
    step_config = runner.env.build_step_config(max_episode_len=runner.episode_len)
    rollout_backend = os.environ.get(
        "SMALLSAT_ROLLOUT_BACKEND",
        getattr(runner.env.env_cfg.control.RL, "rollout_backend", "mjx"),
    )
    if rollout_backend == "freeflyer":
        initial_state = runner.env.freeflyer_state_struct()
        rollout_step_fn = vecenv_step_freeflyer
        rollout_reset_fn = freeflyer_reset_masked
        rollout_state_features_fn = _compute_freeflyer_state_features
    else:
        initial_state = runner.env.state_struct
        rollout_step_fn = None
        rollout_reset_fn = None
        rollout_state_features_fn = None
    initial_state = _with_targeted_initial_offsets(
        initial_state,
        step_config,
        runner.reference_point,
        initial_position_offsets,
        initial_attitude_errors,
    )

    history_len = runner.env.history_len
    state_action_dim = runner.env.obs_dim + runner.env.act_dim
    if runner.env.use_adaptive_approach:
        residual_init = jnp.zeros((num_envs, runner.env.res_dim), dtype=jnp.float32)
        extra = AdaptationRolloutExtra(
            history=jnp.zeros((num_envs, history_len, state_action_dim)),
            counts=jnp.zeros((num_envs,), dtype=jnp.int32),
        )
    else:
        residual_init = jnp.zeros((num_envs, 0), dtype=jnp.float32)
        extra = None

    def _prepare_policy_input(_step, states, residuals, carry_extra):
        return prepare_policy_input_with_residuals(_step, states, residuals, carry_extra)

    def _sample_policy(_step, policy_input, rng_key, carry_extra):
        del _step
        actions = runner.agent.get_control_input("evaluation", policy_input)
        zeros = jnp.zeros((num_envs,), dtype=jnp.float32)
        return actions, zeros, zeros, rng_key, carry_extra

    def _post_step(_step, step_output, actions, residuals, reset_flag, carry_extra):
        del _step
        if not runner.env.use_adaptive_approach:
            return residuals, None, None
        if phase == 1:
            residuals_next = build_adaptive_context(
                commanded_ctrl=step_output.commanded_ctrl,
                applied_ctrl=step_output.applied_ctrl,
                actual_wrench=step_output.actual_wrench,
                desired_wrench=step_output.desired_wrench,
                previous_context=residuals,
                use_adaptive_approach=True,
                adaptive_context_mode=runner.env.adaptive_context_mode,
                thruster_mixer_T=runner.env._thruster_mixer_T,
            )
            return residuals_next, None, carry_extra
        else:
            history, counts, history_full, new_extra = update_history_buffer(
                carry_extra=carry_extra,
                prev_states=step_output.prev_states,
                actions=actions,
                reset_flag=reset_flag,
                history_len=history_len,
            )
            query = build_adaptation_query(
                states=step_output.prev_states,
                desired_wrench=step_output.desired_wrench,
                use_task_conditioned_am=runner.env.use_task_conditioned_am,
            )
            context_pred = runner.adaptation_module(history, query)
            if context_ablation == "zero":
                context_pred = jnp.zeros_like(context_pred)
            elif context_ablation == "shuffled":
                context_pred = jnp.roll(context_pred, shift=1, axis=0)
            residuals_next = jnp.where(history_full[:, None], context_pred, residuals)
            return residuals_next, history_full, new_extra

    rollout = run_functional_rollout(
        step_config=step_config,
        initial_state=initial_state,
        initial_residuals=residual_init,
        rng=runner.agent.key,
        num_steps=runner.episode_len,
        reference_waypoint=runner.reference_point,
        callbacks=FunctionalRolloutCallbacks(
            prepare_policy_input=_prepare_policy_input,
            sample_policy=_sample_policy,
            post_step=_post_step,
            bootstrap_value=make_zero_bootstrap_value(num_envs),
        ),
        extra=extra,
        step_fn=rollout_step_fn,
        reset_fn=rollout_reset_fn,
        state_features_fn=rollout_state_features_fn,
    )
    jax.block_until_ready(rollout.actions)
    runner.agent.key = rollout.final_rng

    step_outputs = rollout.step_outputs
    done_masks = rollout.done_masks
    rewards = step_outputs.rewards
    done_cum = jnp.cumsum(done_masks.astype(jnp.int32), axis=0)
    first_episode_mask = jnp.logical_or(
        done_cum == 0,
        jnp.logical_and(done_masks, done_cum == 1),
    )
    mask_f = first_episode_mask.astype(jnp.float32)
    returns = jnp.sum(rewards * mask_f, axis=0)

    terminals_any = jnp.any(
        jnp.logical_and(step_outputs.success_terminals, first_episode_mask), axis=0
    )
    done_count_by_env = first_episode_mask.astype(jnp.int32).sum(axis=0)
    last_active_idx = jnp.maximum(done_count_by_env - 1, 0)
    env_ids = jnp.arange(num_envs, dtype=jnp.int32)
    final_state = step_outputs.next_states[last_active_idx, env_ids]
    final_pos_error = jnp.linalg.norm(final_state[:, :3], axis=1)
    final_speed = jnp.linalg.norm(final_state[:, 6:9], axis=1)
    final_att_error = jnp.linalg.norm(final_state[:, 3:6], axis=1)
    final_ang_speed = jnp.linalg.norm(final_state[:, 9:12], axis=1)

    pos_ok = final_pos_error <= runner.env.terminal_radius
    speed_ok = final_speed <= runner.env.terminal_max_speed
    att_ok = final_att_error <= runner.env.terminal_max_att_error
    ang_speed_ok = final_ang_speed <= runner.env.terminal_max_ang_speed
    full_pose_ok = jnp.logical_and(pos_ok, att_ok)
    all_conditions_ok = jnp.logical_and(
        full_pose_ok, jnp.logical_and(speed_ok, ang_speed_ok)
    )

    in_set = jnp.logical_and(
        jnp.linalg.norm(step_outputs.next_states[:, :, :3], axis=-1)
        <= runner.env.terminal_radius,
        jnp.linalg.norm(step_outputs.next_states[:, :, 3:6], axis=-1)
        <= runner.env.terminal_max_att_error,
    )
    in_set = jnp.logical_and(in_set, first_episode_mask)

    def _hold_scan(carry, x):
        current, best = carry
        current = jnp.where(x, current + 1, 0)
        best = jnp.maximum(best, current)
        return (current, best), best

    (_, _), best_runs = jax.lax.scan(
        _hold_scan,
        (
            jnp.zeros((num_envs,), dtype=jnp.int32),
            jnp.zeros((num_envs,), dtype=jnp.int32),
        ),
        in_set,
    )
    max_hold_steps = best_runs[-1]

    # Time-averaged first-episode errors and control metrics.
    pos_err_seq = jnp.linalg.norm(step_outputs.next_states[:, :, :3], axis=-1)
    att_err_seq = jnp.linalg.norm(step_outputs.next_states[:, :, 3:6], axis=-1)
    mask_sum = jnp.maximum(jnp.sum(mask_f, axis=0), 1e-6)
    mean_pos_err_by_env = jnp.sum(pos_err_seq * mask_f, axis=0) / mask_sum
    mean_att_err_by_env = jnp.sum(att_err_seq * mask_f, axis=0) / mask_sum

    applied_ctrl = step_outputs.applied_ctrl
    commanded_ctrl = step_outputs.commanded_ctrl
    act_high = jnp.asarray(runner.agent.actor.act_high)[None, None, :]
    act_low = jnp.asarray(runner.agent.actor.act_low)[None, None, :]
    fuel_by_env = jnp.sum(jnp.sum(jnp.abs(applied_ctrl), axis=-1) * mask_f, axis=0)
    active_ctrl = jnp.maximum(jnp.sum(mask_f), 1.0)
    sat_tol = 1e-3
    sat_hi = commanded_ctrl >= (act_high - sat_tol)
    sat_lo = commanded_ctrl <= (act_low + sat_tol)
    sat_any = jnp.logical_or(sat_hi, sat_lo)
    saturation_fraction = jnp.sum(sat_any.astype(jnp.float32) * mask_f[:, :, None]) / (
        active_ctrl * float(commanded_ctrl.shape[-1])
    )

    # Settling time proxy: first step where full-pose set is reached.
    first_set_idx = jnp.argmax(in_set.astype(jnp.int32), axis=0)
    has_set = jnp.any(in_set, axis=0)
    settling_steps = jnp.where(
        has_set,
        first_set_idx.astype(jnp.float32),
        jnp.full_like(first_set_idx, float(runner.episode_len), dtype=jnp.float32),
    )

    final_pos_error_np = np.asarray(jax.device_get(final_pos_error))
    full_pose_success_rate = float(terminals_any.astype(jnp.float32).mean())
    translation_success_rate = float(pos_ok.astype(jnp.float32).mean())
    return {
        "mean_return": float(returns.mean()),
        "success_rate": full_pose_success_rate,
        "full_pose_success_rate": full_pose_success_rate,
        "mean_position_error": float(mean_pos_err_by_env.mean()),
        "mean_attitude_error": float(mean_att_err_by_env.mean()),
        "mean_final_position_error": float(final_pos_error.mean()),
        "mean_final_attitude_error": float(final_att_error.mean()),
        "median_final_position_error": float(np.median(final_pos_error_np)),
        "p75_final_position_error": float(np.percentile(final_pos_error_np, 75.0)),
        "p90_final_position_error": float(np.percentile(final_pos_error_np, 90.0)),
        "fraction_pos_within_radius": float(pos_ok.astype(jnp.float32).mean()),
        "translation_success_rate": translation_success_rate,
        "fraction_full_pose_within_limit": float(
            full_pose_ok.astype(jnp.float32).mean()
        ),
        "fraction_speed_within_limit": float(speed_ok.astype(jnp.float32).mean()),
        "fraction_att_within_limit": float(att_ok.astype(jnp.float32).mean()),
        "fraction_ang_speed_within_limit": float(
            ang_speed_ok.astype(jnp.float32).mean()
        ),
        "fraction_all_conditions_except_hold": float(
            all_conditions_ok.astype(jnp.float32).mean()
        ),
        "mean_consecutive_success_hold_steps": float(
            max_hold_steps.astype(jnp.float32).mean()
        ),
        "settling_time_steps": float(settling_steps.mean()),
        "control_effort": float((fuel_by_env / mask_sum).mean()),
        "fuel": float(fuel_by_env.mean()),
        "saturation_fraction": float(saturation_fraction),
    }


def _context_input_weight_norms(runner: OnPolicyRunner) -> dict[str, float]:
    if not runner.env.use_adaptive_approach:
        return {}
    try:
        state = nnx.state(runner.am)
        leaves = jax.tree_util.tree_leaves(state)
        norms = [
            float(jnp.linalg.norm(leaf))
            for leaf in leaves
            if hasattr(leaf, "shape")
        ]
        if not norms:
            return {}
        return {
            "context_input_weight_norms_l2_mean": float(np.mean(norms)),
            "context_input_weight_norms_l2_max": float(np.max(norms)),
        }
    except Exception:
        return {}


def _bootstrap_ci_mean(
    values: np.ndarray,
    *,
    alpha: float,
    n_bootstrap: int,
    seed: int,
) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        x = float(arr[0])
        return x, x
    n_bootstrap = max(1, int(n_bootstrap))
    alpha = float(np.clip(alpha, 1e-6, 0.999999))
    rng = np.random.default_rng(int(seed))
    sample_idx = rng.integers(0, arr.size, size=(n_bootstrap, arr.size))
    means = arr[sample_idx].mean(axis=1)
    lo = float(np.quantile(means, alpha / 2.0))
    hi = float(np.quantile(means, 1.0 - alpha / 2.0))
    return lo, hi

def _evaluate_scenario(
    runner: OnPolicyRunner,
    table: dict[str, jnp.ndarray],
    *,
    scenario_name: str,
    split_id: int,
    task_feasibility_regime: int | None,
    episodes: int,
    failure_fraction: float,
    disturbance_fraction: float,
    phase: int,
    difficulty_bin: int | None = None,
    failure_type: int | None = None,
    authority_label_any_mask: int | None = None,
    context_ablation: str = "real",
    min_samples_per_split: int = 0,
    enforce_min_samples: bool = False,
    ci_bootstrap_samples: int = 500,
    ci_alpha: float = 0.05,
) -> dict[str, float | str | int | None]:
    cfg = runner.env.env_cfg.control.RL
    failure_start_min = float(getattr(cfg, "curriculum_failure_start_time_min", 0.0))
    failure_start_max = float(getattr(cfg, "curriculum_failure_start_time_max", 0.0))
    disturbance_start_min = float(
        getattr(cfg, "curriculum_disturbance_start_time_min", 0.0)
    )
    disturbance_start_max = float(
        getattr(cfg, "curriculum_disturbance_start_time_max", 0.0)
    )

    metrics_per_episode: list[dict[str, float]] = []
    success_rate_per_episode: list[float] = []
    final_pos_error_per_episode: list[float] = []
    final_att_error_per_episode: list[float] = []
    scenario_payloads: list[dict[str, float]] = []
    for episode_idx in range(episodes):
        print(
            f"[Scenario Eval] {scenario_name}: episode {episode_idx + 1}/{episodes}",
            flush=True,
        )
        runner.env.reset()
        runner.env.reset_perturbations()
        if hasattr(runner.env, "reset_disturbances"):
            runner.env.reset_disturbances()

        key = runner._take_keys()
        perturb_key, disturb_key, failure_start_key, disturbance_start_key = (
            jax.random.split(key, 4)
        )
        failure_start_time = _sample_start_time(
            failure_start_key, failure_start_min, failure_start_max
        )
        task_wrenches = None
        if bool(getattr(cfg, "use_task_conditioned_failure_sampling", True)):
            states = _compute_state_features(
                runner.env.state_struct.mjx_batch,
                runner.reference_point,
            )
            pd_gains = runner.env.env_cfg.control.PD.gains
            task_wrenches = task_wrench_from_state_features(
                states,
                kp_pos=float(getattr(pd_gains, "Kp_x", 0.2)),
                kd_pos=float(getattr(pd_gains, "Kd_x", 1.0)),
                kp_att=float(getattr(pd_gains, "Kp_q", 3.0)),
                kd_att=float(getattr(pd_gains, "Kd_q", 5.0)),
            )
        scenario_result = apply_sampled_failure_scenario_split(
            runner.env,
            key=perturb_key,
            table=table,
            split_id=split_id,
            difficulty_bin=difficulty_bin,
            task_feasibility_regime=task_feasibility_regime,
            failure_type=failure_type,
            authority_label_any_mask=authority_label_any_mask,
            task_wrenches=task_wrenches,
            fraction_perturbed_envs=failure_fraction,
            start_time=failure_start_time,
            return_selection=True,
        )
        scenario_payload, _selected_envs, scenario_indices = scenario_result
        scenario_indices_np = np.asarray(jax.device_get(scenario_indices), dtype=np.int32)
        if scenario_indices_np.size > 0:
            failure_types_np = np.asarray(
                jax.device_get(table["failure_types"])
            )[scenario_indices_np]
            label_mask_np = np.asarray(
                jax.device_get(table["authority_label_mask"])
            )[scenario_indices_np]
            scenario_payload = dict(scenario_payload)
            scenario_payload["scenario/count"] = float(scenario_indices_np.size)
            scenario_payload["scenario/horizon_max_utilization_mean"] = float(
                np.asarray(jax.device_get(table["horizon_max_utilization"]))[
                    scenario_indices_np
                ].mean()
            )
            for ft, name in (
                (0, "stuck_off"),
                (1, "stuck_on"),
                (2, "faulty_valve"),
                (3, "saturated_thrust"),
                (4, "thrust_instability"),
                (5, "constant_disturbance"),
            ):
                scenario_payload[f"scenario/count_failure_{name}"] = float(
                    np.any(failure_types_np == ft, axis=1).sum()
                )
            for bit, name in (
                (LABEL_TORQUE_DEGENERATE, "torque_degenerate"),
                (LABEL_FORCE_DEGENERATE, "force_degenerate"),
                (LABEL_COUPLED_FORCE_TORQUE, "coupled_force_torque"),
                (LABEL_SATURATION_PRONE, "saturation_prone"),
                (LABEL_NEAR_DEPENDENT, "near_dependent"),
                (LABEL_BIAS_DOMINATED, "bias_dominated"),
                (LABEL_NONLINEAR_MISMATCH, "nonlinear_mismatch"),
            ):
                scenario_payload[f"scenario/count_label_{name}"] = float(
                    ((label_mask_np & int(bit)) != 0).sum()
                )
        scenario_payloads.append(scenario_payload)

        if disturbance_fraction > 0.0:
            disturbance_start_time = _sample_start_time(
                disturbance_start_key, disturbance_start_min, disturbance_start_max
            )
            runner.env.apply_random_disturbance(
                key=disturb_key,
                fraction_disturbed_envs=disturbance_fraction,
                start_time=disturbance_start_time,
            )

        metrics = _rollout_once(runner, phase=phase, context_ablation=context_ablation)
        metrics_per_episode.append(metrics)
        success_rate_per_episode.append(float(metrics["success_rate"]))
        final_pos_error_per_episode.append(float(metrics["mean_final_position_error"]))
        final_att_error_per_episode.append(float(metrics["mean_final_attitude_error"]))

    aggregate: dict[str, float | str | int | None] = {
        "scenario": scenario_name,
        "split_id": split_id,
        "task_feasibility_regime": task_feasibility_regime,
        "difficulty_bin": difficulty_bin,
        "failure_type": failure_type,
        "episodes": episodes,
        "failure_fraction": failure_fraction,
        "disturbance_fraction": disturbance_fraction,
        "context_ablation": context_ablation,
    }
    metric_keys = metrics_per_episode[0].keys()
    for key in metric_keys:
        aggregate[key] = float(np.mean([episode[key] for episode in metrics_per_episode]))
    for key in scenario_payloads[0].keys():
        aggregate[key.replace("/", "_")] = float(
            np.mean([payload[key] for payload in scenario_payloads])
        )
    sampled_count = int(round(float(aggregate.get("scenario_count", 0.0))))
    aggregate["sample_count_ok"] = int(sampled_count >= int(min_samples_per_split))
    aggregate["sample_count_threshold"] = int(min_samples_per_split)
    if enforce_min_samples and sampled_count < int(min_samples_per_split):
        raise RuntimeError(
            f"{scenario_name}: sampled {sampled_count} < required {int(min_samples_per_split)}"
        )
    sr_lo, sr_hi = _bootstrap_ci_mean(
        np.asarray(success_rate_per_episode, dtype=np.float64),
        alpha=ci_alpha,
        n_bootstrap=ci_bootstrap_samples,
        seed=20260527,
    )
    pos_lo, pos_hi = _bootstrap_ci_mean(
        np.asarray(final_pos_error_per_episode, dtype=np.float64),
        alpha=ci_alpha,
        n_bootstrap=ci_bootstrap_samples,
        seed=20260528,
    )
    att_lo, att_hi = _bootstrap_ci_mean(
        np.asarray(final_att_error_per_episode, dtype=np.float64),
        alpha=ci_alpha,
        n_bootstrap=ci_bootstrap_samples,
        seed=20260529,
    )
    aggregate["success_rate_ci_low"] = sr_lo
    aggregate["success_rate_ci_high"] = sr_hi
    aggregate["mean_final_position_error_ci_low"] = pos_lo
    aggregate["mean_final_position_error_ci_high"] = pos_hi
    aggregate["mean_final_attitude_error_ci_low"] = att_lo
    aggregate["mean_final_attitude_error_ci_high"] = att_hi
    aggregate.update(_context_input_weight_norms(runner))
    return aggregate


def _evaluate_targeted_scenario(
    runner: OnPolicyRunner,
    table: dict[str, jnp.ndarray],
    *,
    scenario_name: str,
    split_id: int,
    task_feasibility_regime: int | None,
    episodes: int,
    failure_fraction: float,
    start_distance: float,
    phase: int,
    context_ablation: str = "real",
    min_samples_per_split: int = 0,
    enforce_min_samples: bool = False,
    ci_bootstrap_samples: int = 500,
    ci_alpha: float = 0.05,
) -> dict[str, float | str | int | None]:
    num_perturbed = int(
        runner.env.num_envs * max(0.0, min(1.0, failure_fraction))
    )
    if num_perturbed <= 0:
        raise ValueError("targeted failure_fraction must select at least one env")

    metrics_per_episode: list[dict[str, float]] = []
    success_rate_per_episode: list[float] = []
    final_pos_error_per_episode: list[float] = []
    final_att_error_per_episode: list[float] = []
    scenario_payloads: list[dict[str, float]] = []
    for episode_idx in range(episodes):
        print(
            f"[Scenario Eval] {scenario_name}: episode {episode_idx + 1}/{episodes}",
            flush=True,
        )
        runner.env.reset()
        runner.env.reset_perturbations()
        if hasattr(runner.env, "reset_disturbances"):
            runner.env.reset_disturbances()

        key = runner._take_keys()
        scenario_key, apply_key = jax.random.split(key, 2)
        selected_envs = jnp.arange(num_perturbed, dtype=jnp.int32)
        scenario_indices = sample_scenario_indices(
            scenario_key,
            table,
            split_id=split_id,
            count=num_perturbed,
            task_feasibility_regime=task_feasibility_regime,
        )
        apply_failure_scenarios(
            runner.env,
            key=apply_key,
            table=table,
            selected_envs=selected_envs,
            scenario_indices=scenario_indices,
            start_time=0.0,
        )
        scenario_payloads.append(
            scenario_selection_payload(table, scenario_indices, prefix="scenario")
        )
        scenario_indices_np = np.asarray(jax.device_get(scenario_indices), dtype=np.int32)
        if scenario_indices_np.size > 0:
            failure_types_np = np.asarray(
                jax.device_get(table["failure_types"])
            )[scenario_indices_np]
            label_mask_np = np.asarray(
                jax.device_get(table["authority_label_mask"])
            )[scenario_indices_np]
            payload = dict(scenario_payloads[-1])
            payload["scenario/count"] = float(scenario_indices_np.size)
            payload["scenario/horizon_max_utilization_mean"] = float(
                np.asarray(jax.device_get(table["horizon_max_utilization"]))[
                    scenario_indices_np
                ].mean()
            )
            for ft, name in (
                (0, "stuck_off"),
                (1, "stuck_on"),
                (2, "faulty_valve"),
                (3, "saturated_thrust"),
                (4, "thrust_instability"),
                (5, "constant_disturbance"),
            ):
                payload[f"scenario/count_failure_{name}"] = float(
                    np.any(failure_types_np == ft, axis=1).sum()
                )
            for bit, name in (
                (LABEL_TORQUE_DEGENERATE, "torque_degenerate"),
                (LABEL_FORCE_DEGENERATE, "force_degenerate"),
                (LABEL_COUPLED_FORCE_TORQUE, "coupled_force_torque"),
                (LABEL_SATURATION_PRONE, "saturation_prone"),
                (LABEL_NEAR_DEPENDENT, "near_dependent"),
                (LABEL_BIAS_DOMINATED, "bias_dominated"),
                (LABEL_NONLINEAR_MISMATCH, "nonlinear_mismatch"),
            ):
                payload[f"scenario/count_label_{name}"] = float(
                    ((label_mask_np & int(bit)) != 0).sum()
                )
            scenario_payloads[-1] = payload

        offsets, attitude_errors = targeted_pose_errors_from_scenarios(
            table,
            scenario_indices,
            distance=start_distance,
        )
        if num_perturbed < runner.env.num_envs:
            offsets = jnp.concatenate(
                [
                    offsets,
                    jnp.zeros(
                        (runner.env.num_envs - num_perturbed, 3),
                        dtype=offsets.dtype,
                    ),
                ],
                axis=0,
            )
            attitude_errors = jnp.concatenate(
                [
                    attitude_errors,
                    jnp.zeros(
                        (runner.env.num_envs - num_perturbed, 3),
                        dtype=attitude_errors.dtype,
                    ),
                ],
                axis=0,
            )
        metrics = _rollout_once(
            runner,
            initial_position_offsets=offsets,
            initial_attitude_errors=attitude_errors,
            phase=phase,
            context_ablation=context_ablation,
        )
        metrics_per_episode.append(metrics)
        success_rate_per_episode.append(float(metrics["success_rate"]))
        final_pos_error_per_episode.append(float(metrics["mean_final_position_error"]))
        final_att_error_per_episode.append(float(metrics["mean_final_attitude_error"]))

    aggregate: dict[str, float | str | int | None] = {
        "scenario": scenario_name,
        "split_id": split_id,
        "task_feasibility_regime": task_feasibility_regime,
        "episodes": episodes,
        "failure_fraction": failure_fraction,
        "disturbance_fraction": 0.0,
        "targeted_start_distance": start_distance,
        "context_ablation": context_ablation,
    }
    metric_keys = metrics_per_episode[0].keys()
    for key in metric_keys:
        aggregate[key] = float(np.mean([episode[key] for episode in metrics_per_episode]))
    for key in scenario_payloads[0].keys():
        aggregate[key.replace("/", "_")] = float(
            np.mean([payload[key] for payload in scenario_payloads])
        )
    sampled_count = int(round(float(aggregate.get("scenario_count", 0.0))))
    aggregate["sample_count_ok"] = int(sampled_count >= int(min_samples_per_split))
    aggregate["sample_count_threshold"] = int(min_samples_per_split)
    if enforce_min_samples and sampled_count < int(min_samples_per_split):
        raise RuntimeError(
            f"{scenario_name}: sampled {sampled_count} < required {int(min_samples_per_split)}"
        )
    sr_lo, sr_hi = _bootstrap_ci_mean(
        np.asarray(success_rate_per_episode, dtype=np.float64),
        alpha=ci_alpha,
        n_bootstrap=ci_bootstrap_samples,
        seed=20260627,
    )
    pos_lo, pos_hi = _bootstrap_ci_mean(
        np.asarray(final_pos_error_per_episode, dtype=np.float64),
        alpha=ci_alpha,
        n_bootstrap=ci_bootstrap_samples,
        seed=20260628,
    )
    att_lo, att_hi = _bootstrap_ci_mean(
        np.asarray(final_att_error_per_episode, dtype=np.float64),
        alpha=ci_alpha,
        n_bootstrap=ci_bootstrap_samples,
        seed=20260629,
    )
    aggregate["success_rate_ci_low"] = sr_lo
    aggregate["success_rate_ci_high"] = sr_hi
    aggregate["mean_final_position_error_ci_low"] = pos_lo
    aggregate["mean_final_position_error_ci_high"] = pos_hi
    aggregate["mean_final_attitude_error_ci_low"] = att_lo
    aggregate["mean_final_attitude_error_ci_high"] = att_hi
    aggregate.update(_context_input_weight_norms(runner))
    return aggregate


def main() -> None:
    args = _parse_args()
    env = AstrobeeEnvVectorized(
        args=_sim_args(args),
        run_name=args.run_name,
        train_with_failures=True,
        use_pretrained=False,
        use_adaptive_approach=args.use_adaptive_approach,
        am_architecture=args.am_architecture,
        adaptive_context_mode=args.adaptive_context_mode,
        use_task_conditioned_am=args.task_conditioned,
        am_predict_delta_weight=args.predict_delta_weight,
        am_predict_tracking_weight=args.predict_tracking_weight,
        am_predict_authority_weight=args.predict_authority_weight,
        num_envs=args.num_envs,
    )
    planner = OraclePlannerRL(env, radius=0.0)
    runner = OnPolicyRunner(env, planner)
    ckpt_name = _load_modules(runner, args.checkpoint, int(args.phase))
    print(
        "[Scenario Eval] Loaded checkpoint: "
        f"{ckpt_name} | adaptive={env.use_adaptive_approach} "
        f"context={env.adaptive_context_mode} phase={args.phase}",
        flush=True,
    )

    cfg = env.env_cfg.control.RL
    table = build_failure_scenario_table(
        env._thruster_mixer_T,
        runner.agent.actor.act_low,
        runner.agent.actor.act_high,
        max_faults=int(getattr(cfg, "failure_scenario_max_faults", 12)),
        exhaustive_faults=int(
            getattr(cfg, "failure_scenario_exhaustive_faults", 2)
        ),
        sampled_per_fault_count=int(
            getattr(cfg, "failure_scenario_sampled_per_fault_count", 512)
        ),
        min_rank=int(getattr(cfg, "failure_scenario_min_rank", 6)),
        stress_quantile=float(getattr(cfg, "failure_scenario_stress_quantile", 0.9)),
        mild_effectiveness=float(
            getattr(cfg, "failure_scenario_mild_effectiveness", 0.5)
        ),
    )
    print(
        "[Scenario Eval] Table counts "
        f"splits={scenario_split_counts(table)} "
        f"train_task_regimes={scenario_task_regime_counts(table, SPLIT_TRAIN)}",
        flush=True,
    )

    rows = []
    fixed_evals = [
        ("nominal", None, None, None, 0.0),
        ("easy_feasible", TASK_REGIME_EASY_FEASIBLE, None, None, float(args.failure_fraction)),
        ("hard_feasible", TASK_REGIME_HARD_FEASIBLE, None, None, float(args.failure_fraction)),
        ("near_infeasible", TASK_REGIME_NEAR_INFEASIBLE, None, None, float(args.failure_fraction)),
        ("stuck_off_only", None, 0, None, float(args.failure_fraction)),
        ("stuck_on_only", None, 1, None, float(args.failure_fraction)),
        ("nonlinear_mismatch", None, None, LABEL_NONLINEAR_MISMATCH, float(args.failure_fraction)),
        ("constant_disturbance", None, 5, None, float(args.failure_fraction)),
        ("torque_degenerate", None, None, LABEL_TORQUE_DEGENERATE, float(args.failure_fraction)),
        ("force_degenerate", None, None, LABEL_FORCE_DEGENERATE, float(args.failure_fraction)),
        ("coupled_force_torque", None, None, LABEL_COUPLED_FORCE_TORQUE, float(args.failure_fraction)),
        ("saturation_prone", None, None, LABEL_SATURATION_PRONE, float(args.failure_fraction)),
        ("near_dependent", None, None, LABEL_NEAR_DEPENDENT, float(args.failure_fraction)),
        ("bias_dominated", None, None, LABEL_BIAS_DOMINATED, float(args.failure_fraction)),
    ]
    if args.include_infeasible:
        fixed_evals.append(
            ("infeasible_stress", TASK_REGIME_INFEASIBLE, None, None, float(args.failure_fraction))
        )

    context_modes = (
        ("real", "zero", "shuffled")
        if args.context_ablation == "all"
        else (args.context_ablation,)
    )
    for context_mode in context_modes:
        for scenario_name, task_regime, failure_type, label_mask, frac in fixed_evals:
            if frac > 0.0 and not _scenario_filter_has_rows(
                table,
                split_id=SPLIT_TRAIN,
                task_feasibility_regime=task_regime,
                failure_type=failure_type,
                authority_label_any_mask=label_mask,
            ):
                print(f"[Scenario Eval] Skipping {scenario_name}: no matching scenarios")
                continue
            rows.append(
                _evaluate_scenario(
                    runner,
                    table,
                    scenario_name=scenario_name,
                    split_id=SPLIT_TRAIN,
                    task_feasibility_regime=task_regime,
                    episodes=max(1, int(args.episodes)),
                    failure_fraction=frac,
                    disturbance_fraction=0.0,
                    phase=int(args.phase),
                    failure_type=failure_type,
                    authority_label_any_mask=label_mask,
                    context_ablation=context_mode,
                    min_samples_per_split=int(args.min_samples_per_split),
                    enforce_min_samples=bool(args.enforce_min_samples),
                    ci_bootstrap_samples=int(args.ci_bootstrap_samples),
                    ci_alpha=float(args.ci_alpha),
                )
            )

    targeted_task_evals = [
        ("targeted_easy_feasible", SPLIT_TRAIN, TASK_REGIME_EASY_FEASIBLE),
        ("targeted_hard_feasible", SPLIT_TRAIN, TASK_REGIME_HARD_FEASIBLE),
        ("targeted_near_infeasible", SPLIT_TRAIN, TASK_REGIME_NEAR_INFEASIBLE),
    ]
    if args.include_infeasible:
        targeted_task_evals.append(
            ("targeted_infeasible_stress", SPLIT_TRAIN, TASK_REGIME_INFEASIBLE)
        )
    if not args.skip_targeted:
        for context_mode in context_modes:
            for scenario_name, split_id, task_regime in targeted_task_evals:
                if not _scenario_filter_has_rows(
                    table,
                    split_id=split_id,
                    task_feasibility_regime=task_regime,
                ):
                    print(f"[Scenario Eval] Skipping {scenario_name}: no matching scenarios")
                    continue
                rows.append(
                    _evaluate_targeted_scenario(
                        runner,
                        table,
                        scenario_name=scenario_name,
                        split_id=split_id,
                        task_feasibility_regime=task_regime,
                        episodes=max(1, int(args.episodes)),
                        failure_fraction=float(args.targeted_failure_fraction),
                        start_distance=float(args.targeted_start_distance),
                        phase=int(args.phase),
                        context_ablation=context_mode,
                        min_samples_per_split=int(args.min_samples_per_split),
                        enforce_min_samples=bool(args.enforce_min_samples),
                        ci_bootstrap_samples=int(args.ci_bootstrap_samples),
                        ci_alpha=float(args.ci_alpha),
                    )
                )

    output_path = Path(
        args.output
        or f"experiments/rl_results/scenario_split_eval/{args.run_name}_scenario_split_eval.csv"
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = list(dict.fromkeys(key for row in rows for key in row.keys()))
    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"\n[Scenario Eval] Wrote {output_path}")
    for row in rows:
        print(
            f"{row['scenario']:>12s}: full_pose={row['success_rate']:.3f}, "
            f"translation={row['translation_success_rate']:.3f}, "
            f"return={row['mean_return']:.3f}, "
            f"p90_pos={row['p90_final_position_error']:.3f}, "
            f"hold_steps={row['mean_consecutive_success_hold_steps']:.2f}"
        )
        if args.wandb and wandb.run is not None:
            prefix = f"scenario_split_eval/{row['scenario']}"
            wandb.log(
                {
                    f"{prefix}/{key}": value
                    for key, value in row.items()
                    if isinstance(value, (float, int))
                }
            )
    # Main adaptive claim gate:
    # real_context > zero_context and shuffled_context on hard/near-boundary feasible splits.
    if args.context_ablation == "all":
        compare_scenarios = {"hard_feasible", "near_infeasible", "targeted_hard_feasible", "targeted_near_infeasible"}
        by_key = {}
        for row in rows:
            scenario = str(row.get("scenario"))
            context = str(row.get("context_ablation"))
            if scenario in compare_scenarios:
                by_key[(scenario, context)] = float(row.get("success_rate", 0.0))
        for scenario in sorted(compare_scenarios):
            real = by_key.get((scenario, "real"))
            zero = by_key.get((scenario, "zero"))
            shuffled = by_key.get((scenario, "shuffled"))
            if real is None or zero is None or shuffled is None:
                continue
            claim_ok = bool(real > zero and real > shuffled)
            print(
                f"[Scenario Eval][Claim] {scenario}: real={real:.3f}, zero={zero:.3f}, "
                f"shuffled={shuffled:.3f}, real_gt_both={claim_ok}",
                flush=True,
            )
            delta_real_zero = real - zero
            delta_real_shuffled = real - shuffled
            print(
                f"[Scenario Eval][ClaimDelta] {scenario}: "
                f"real_minus_zero={delta_real_zero:.3f}, "
                f"real_minus_shuffled={delta_real_shuffled:.3f}",
                flush=True,
            )
    if args.wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
