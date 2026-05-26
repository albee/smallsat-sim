"""Evaluate a trained policy on task-feasibility failure-scenario splits."""

from __future__ import annotations

import argparse
import csv
import os
from argparse import Namespace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import wandb
from flax import nnx
from mujoco import mjx

from smallsat_sim.controllers.rl.runners.failure_scenarios import (
    BIN_NAMES,
    FAILURE_TYPE_NAMES,
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


TASK_FEASIBLE_SCENARIO_EVALS = (
    ("task_easy_feasible", SPLIT_TRAIN, TASK_REGIME_EASY_FEASIBLE),
    ("task_hard_feasible", SPLIT_TRAIN, TASK_REGIME_HARD_FEASIBLE),
    ("task_near_infeasible", SPLIT_TRAIN, TASK_REGIME_NEAR_INFEASIBLE),
    ("task_infeasible_stress", SPLIT_TRAIN, TASK_REGIME_INFEASIBLE),
)

TARGETED_TASK_FEASIBLE_SCENARIO_EVALS = (
    ("targeted_easy_feasible", SPLIT_TRAIN, TASK_REGIME_EASY_FEASIBLE),
    ("targeted_hard_feasible", SPLIT_TRAIN, TASK_REGIME_HARD_FEASIBLE),
    ("targeted_near_infeasible", SPLIT_TRAIN, TASK_REGIME_NEAR_INFEASIBLE),
    ("targeted_infeasible_stress", SPLIT_TRAIN, TASK_REGIME_INFEASIBLE),
)


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
    return bool(jax.device_get(jnp.any(mask)))


def _rollout_once(
    runner: OnPolicyRunner,
    *,
    initial_position_offsets: jnp.ndarray | None = None,
    initial_attitude_errors: jnp.ndarray | None = None,
    phase: int = 1,
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

    final_pos_error_np = np.asarray(jax.device_get(final_pos_error))
    full_pose_success_rate = float(terminals_any.astype(jnp.float32).mean())
    translation_success_rate = float(pos_ok.astype(jnp.float32).mean())
    return {
        "mean_return": float(returns.mean()),
        "success_rate": full_pose_success_rate,
        "full_pose_success_rate": full_pose_success_rate,
        "mean_final_position_error": float(final_pos_error.mean()),
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
    }

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
        scenario_payload = apply_sampled_failure_scenario_split(
            runner.env,
            key=perturb_key,
            table=table,
            split_id=split_id,
            difficulty_bin=difficulty_bin,
            task_feasibility_regime=task_feasibility_regime,
            failure_type=failure_type,
            task_wrenches=task_wrenches,
            fraction_perturbed_envs=failure_fraction,
            start_time=failure_start_time,
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

        metrics_per_episode.append(_rollout_once(runner, phase=phase))

    aggregate: dict[str, float | str | int | None] = {
        "scenario": scenario_name,
        "split_id": split_id,
        "task_feasibility_regime": task_feasibility_regime,
        "difficulty_bin": difficulty_bin,
        "failure_type": failure_type,
        "episodes": episodes,
        "failure_fraction": failure_fraction,
        "disturbance_fraction": disturbance_fraction,
    }
    metric_keys = metrics_per_episode[0].keys()
    for key in metric_keys:
        aggregate[key] = float(np.mean([episode[key] for episode in metrics_per_episode]))
    for key in scenario_payloads[0].keys():
        aggregate[key.replace("/", "_")] = float(
            np.mean([payload[key] for payload in scenario_payloads])
        )
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
) -> dict[str, float | str | int | None]:
    num_perturbed = int(
        runner.env.num_envs * max(0.0, min(1.0, failure_fraction))
    )
    if num_perturbed <= 0:
        raise ValueError("targeted failure_fraction must select at least one env")

    metrics_per_episode: list[dict[str, float]] = []
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
        metrics_per_episode.append(
            _rollout_once(
                runner,
                initial_position_offsets=offsets,
                initial_attitude_errors=attitude_errors,
                phase=phase,
            )
        )

    aggregate: dict[str, float | str | int | None] = {
        "scenario": scenario_name,
        "split_id": split_id,
        "task_feasibility_regime": task_feasibility_regime,
        "episodes": episodes,
        "failure_fraction": failure_fraction,
        "disturbance_fraction": 0.0,
        "targeted_start_distance": start_distance,
    }
    metric_keys = metrics_per_episode[0].keys()
    for key in metric_keys:
        aggregate[key] = float(np.mean([episode[key] for episode in metrics_per_episode]))
    for key in scenario_payloads[0].keys():
        aggregate[key.replace("/", "_")] = float(
            np.mean([payload[key] for payload in scenario_payloads])
        )
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
    for scenario_name, split_id, task_regime in TASK_FEASIBLE_SCENARIO_EVALS:
        if not _scenario_filter_has_rows(
            table,
            split_id=split_id,
            task_feasibility_regime=task_regime,
        ):
            print(f"[Scenario Eval] Skipping {scenario_name}: no matching scenarios")
            continue
        rows.append(
            _evaluate_scenario(
                runner,
                table,
                scenario_name=scenario_name,
                split_id=split_id,
                task_feasibility_regime=task_regime,
                episodes=max(1, int(args.episodes)),
                failure_fraction=float(args.failure_fraction),
                disturbance_fraction=float(args.disturbance_fraction),
                phase=int(args.phase),
            )
        )
    for difficulty_bin, difficulty_name in BIN_NAMES.items():
        scenario_name = f"difficulty_{difficulty_name}"
        if not _scenario_filter_has_rows(
            table,
            split_id=SPLIT_TRAIN,
            difficulty_bin=difficulty_bin,
        ):
            print(f"[Scenario Eval] Skipping {scenario_name}: no matching scenarios")
            continue
        rows.append(
            _evaluate_scenario(
                runner,
                table,
                scenario_name=scenario_name,
                split_id=SPLIT_TRAIN,
                task_feasibility_regime=None,
                difficulty_bin=difficulty_bin,
                episodes=max(1, int(args.episodes)),
                failure_fraction=float(args.failure_fraction),
                disturbance_fraction=float(args.disturbance_fraction),
                phase=int(args.phase),
            )
        )
    for failure_type, failure_name in FAILURE_TYPE_NAMES.items():
        scenario_name = f"failure_{failure_name}"
        if not _scenario_filter_has_rows(
            table,
            split_id=SPLIT_TRAIN,
            failure_type=failure_type,
        ):
            print(f"[Scenario Eval] Skipping {scenario_name}: no matching scenarios")
            continue
        rows.append(
            _evaluate_scenario(
                runner,
                table,
                scenario_name=scenario_name,
                split_id=SPLIT_TRAIN,
                task_feasibility_regime=None,
                failure_type=failure_type,
                episodes=max(1, int(args.episodes)),
                failure_fraction=float(args.failure_fraction),
                disturbance_fraction=float(args.disturbance_fraction),
                phase=int(args.phase),
            )
        )
    if not args.skip_targeted:
        for scenario_name, split_id, task_regime in TARGETED_TASK_FEASIBLE_SCENARIO_EVALS:
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
    if args.wandb and wandb.run is not None:
        wandb.finish()


if __name__ == "__main__":
    main()
