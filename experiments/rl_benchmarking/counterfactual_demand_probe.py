"""
Counterfactual demand probing for task-conditioned adaptation modules.

This diagnostic answers a narrow question: for the same closed-loop history, does
the adaptation latent change when the current desired wrench/query changes?
It is intended for post-training analysis and paper figures, not for PPO training.
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
from flax import nnx

from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.controllers.rl.runners.runner_utils import (
    load_trained_modules,
    load_training_data,
)
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.utils.helpers import get_args


DEFAULT_RUN_NAME = "ppo_adaptive_cross_attention_task_predictive"
DEFAULT_NUM_SAMPLES = 512
DEFAULT_DIAGNOSTIC_ENVS = 128


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Probe task-conditioned adaptation under counterfactual wrench queries.",
        parents=[argparse.ArgumentParser(add_help=False)],
    )
    parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    parser.add_argument(
        "--am-architecture",
        default="transformer_cross_attention",
        choices=("transformer", "transformer_cross_attention", "cnn"),
    )
    parser.add_argument(
        "--adaptive-context-mode",
        default="structured",
        choices=(
            "residual",
            "residual_effectiveness",
            "residual_controllability",
            "structured",
        ),
    )
    parser.add_argument("--task-conditioned", action="store_true", default=True)
    parser.add_argument(
        "--no-task-conditioned", dest="task_conditioned", action="store_false"
    )
    parser.add_argument("--predict-delta-weight", type=float, default=0.1)
    parser.add_argument("--predict-tracking-weight", type=float, default=0.1)
    parser.add_argument("--predict-authority-weight", type=float, default=0.1)
    parser.add_argument("--num-samples", type=int, default=DEFAULT_NUM_SAMPLES)
    parser.add_argument("--diagnostic-envs", type=int, default=DEFAULT_DIAGNOSTIC_ENVS)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--wrench-scale",
        type=float,
        default=None,
        help="Counterfactual wrench magnitude. Defaults to median commanded wrench norm.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("experiments/rl_results/counterfactual_demand_probe"),
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Also log summary metrics to wandb.",
    )
    known, passthrough = parser.parse_known_args()

    # Reuse the repo's common args object for environment construction.
    common_args = get_args_from_passthrough(passthrough)
    known.common_args = common_args
    return known


def get_args_from_passthrough(passthrough: list[str]) -> argparse.Namespace:
    import sys

    old_argv = sys.argv
    try:
        sys.argv = [old_argv[0], *passthrough]
        args = get_args()
    finally:
        sys.argv = old_argv
    return args


def _load_runner(args: argparse.Namespace) -> OnPolicyRunner:
    env = AstrobeeEnvVectorized(
        args=args.common_args,
        run_name=args.run_name,
        train_with_failures=True,
        use_pretrained=False,
        use_adaptive_approach=True,
        am_architecture=args.am_architecture,
        adaptive_context_mode=args.adaptive_context_mode,
        use_task_conditioned_am=args.task_conditioned,
        am_predict_delta_weight=args.predict_delta_weight,
        am_predict_tracking_weight=args.predict_tracking_weight,
        am_predict_authority_weight=args.predict_authority_weight,
        num_envs=args.diagnostic_envs,
    )
    planner = OraclePlannerRL(env, radius=0.0)
    runner = OnPolicyRunner(env, planner)

    am_path = Path(runner.ckpt_dir) / runner.adaptation_module_file_name
    if not am_path.exists():
        raise FileNotFoundError(
            f"Missing adaptation-module checkpoint: {am_path}. "
            "Train the matching adaptive run before probing."
        )
    am_state = load_trained_modules(runner.ckpt_dir, runner.adaptation_module_file_name)[
        "am_model"
    ]
    nnx.update(runner.am, am_state)
    return runner


def _load_training_rollout(runner: OnPolicyRunner) -> dict[str, jnp.ndarray]:
    data_path = Path(runner.ckpt_dir) / runner.training_data_file_name
    if not data_path.exists():
        raise FileNotFoundError(
            f"Missing rollout data checkpoint: {data_path}. "
            "Run the matching training script through the final epoch first."
        )
    return load_training_data(runner.ckpt_dir, runner.training_data_file_name)


def _sample_history_windows(
    data: dict[str, jnp.ndarray],
    *,
    history_len: int,
    max_samples: int,
    seed: int,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    obs = np.asarray(data["obs"])
    act = np.asarray(data["act"])
    done = np.asarray(data.get("done", np.zeros(obs.shape[:2], dtype=bool)))
    num_steps, num_envs, obs_dim = obs.shape
    act_dim = act.shape[-1]
    if num_steps < history_len:
        raise ValueError(
            f"Need at least {history_len} rollout steps, got {num_steps}."
        )

    rng = np.random.default_rng(seed)
    features = np.concatenate([obs, act], axis=-1)
    histories = []
    states = []
    policy_actions = []
    attempts = 0
    max_attempts = max_samples * 100
    while len(histories) < max_samples and attempts < max_attempts:
        attempts += 1
        env_idx = int(rng.integers(0, num_envs))
        t = int(rng.integers(history_len - 1, num_steps))
        start = t - history_len + 1
        if done[start:t, env_idx].any():
            continue
        histories.append(features[start : t + 1, env_idx, :])
        states.append(obs[t, env_idx, :])
        policy_actions.append(act[t, env_idx, :])

    if not histories:
        raise RuntimeError("Could not sample any valid history windows.")

    if len(histories) < max_samples:
        print(
            f"[Counterfactual Probe] sampled {len(histories)} valid windows "
            f"after {attempts} attempts; requested {max_samples}."
        )

    return (
        jnp.asarray(np.stack(histories).astype(np.float32)),
        jnp.asarray(np.stack(states).astype(np.float32)),
        jnp.asarray(np.stack(policy_actions).astype(np.float32)),
    )


def _candidate_wrenches(
    policy_actions: jnp.ndarray,
    thruster_mixer_T: jnp.ndarray,
    wrench_scale: float | None,
) -> tuple[list[str], jnp.ndarray]:
    empirical = policy_actions @ thruster_mixer_T
    norms = jnp.linalg.norm(empirical, axis=1)
    nonzero_norms = norms[norms > 1e-6]
    if wrench_scale is None:
        scale = float(jnp.median(nonzero_norms)) if nonzero_norms.size else 0.1
        scale = max(scale, 0.05)
    else:
        scale = float(wrench_scale)

    basis = jnp.eye(6, dtype=jnp.float32)
    candidates = [jnp.zeros((6,), dtype=jnp.float32)]
    labels = ["zero"]
    for idx, axis in enumerate(("fx", "fy", "fz", "tx", "ty", "tz")):
        candidates.append(scale * basis[idx])
        labels.append(f"+{axis}")
        candidates.append(-scale * basis[idx])
        labels.append(f"-{axis}")

    return labels, jnp.stack(candidates, axis=0)


def _probe_module(
    runner: OnPolicyRunner,
    histories: jnp.ndarray,
    states: jnp.ndarray,
    candidate_wrenches: jnp.ndarray,
) -> dict[str, jnp.ndarray]:
    num_samples = histories.shape[0]
    num_candidates = candidate_wrenches.shape[0]
    histories_flat = jnp.repeat(histories, num_candidates, axis=0)
    states_flat = jnp.repeat(states, num_candidates, axis=0)
    wrenches_flat = jnp.tile(candidate_wrenches, (num_samples, 1))
    queries_flat = jnp.concatenate([states_flat, wrenches_flat], axis=1)

    if runner.env.am_architecture == "transformer_cross_attention":

        def _apply(hist, query):
            mu, log_sigma, delta, tracking, authority, attention = runner.am(
                hist,
                query,
                return_stats=True,
                return_predictions=True,
                return_attention=True,
            )
            return mu, log_sigma, delta, tracking, authority, attention

        mu, log_sigma, delta, tracking, authority, attention = jax.vmap(_apply)(
            histories_flat, queries_flat
        )
    elif runner.env.am_architecture == "transformer":

        def _apply(hist, query):
            mu, log_sigma, delta, tracking, authority = runner.am(
                hist,
                query,
                return_stats=True,
                return_predictions=True,
            )
            return mu, log_sigma, delta, tracking, authority

        mu, log_sigma, delta, tracking, authority = jax.vmap(_apply)(
            histories_flat, queries_flat
        )
        attention = jnp.zeros(
            (num_samples * num_candidates, 1, runner.env.history_len),
            dtype=mu.dtype,
        )
    else:

        def _apply(hist, query):
            mu, delta, tracking, authority = runner.am(
                hist,
                query,
                return_predictions=True,
            )
            log_sigma = jnp.zeros_like(mu)
            return mu, log_sigma, delta, tracking, authority

        mu, log_sigma, delta, tracking, authority = jax.vmap(_apply)(
            histories_flat, queries_flat
        )
        attention = jnp.zeros(
            (num_samples * num_candidates, 1, runner.env.history_len),
            dtype=mu.dtype,
        )

    shape_prefix = (num_samples, num_candidates)
    return {
        "mu": mu.reshape(*shape_prefix, -1),
        "log_sigma": log_sigma.reshape(*shape_prefix, -1),
        "delta": delta.reshape(*shape_prefix, -1),
        "tracking": tracking.reshape(*shape_prefix, -1),
        "authority": authority.reshape(*shape_prefix, -1),
        "attention": attention.reshape(*shape_prefix, attention.shape[-2], attention.shape[-1]),
    }


def _summary_metrics(outputs: dict[str, jnp.ndarray]) -> dict[str, float]:
    mu = outputs["mu"]
    attention = outputs["attention"]
    tracking = outputs["tracking"]
    authority = outputs["authority"]

    zero_mu = mu[:, :1, :]
    latent_delta = jnp.linalg.norm(mu - zero_mu, axis=-1)
    latent_std = jnp.mean(jnp.std(mu, axis=1), axis=-1)
    latent_norm = jnp.linalg.norm(mu, axis=-1)

    attn_mean_heads = attention.mean(axis=2)
    zero_attn = attn_mean_heads[:, :1, :]
    attention_l1_from_zero = jnp.sum(jnp.abs(attn_mean_heads - zero_attn), axis=-1)
    attention_entropy = -jnp.sum(
        attn_mean_heads * jnp.log(attn_mean_heads + 1e-8), axis=-1
    )
    max_entropy = jnp.log(jnp.asarray(attn_mean_heads.shape[-1], dtype=jnp.float32))
    time_axis = jnp.linspace(0.0, 1.0, attn_mean_heads.shape[-1])
    recency_center = jnp.sum(attn_mean_heads * time_axis, axis=-1)

    summary = {
        "latent_delta_from_zero_mean": float(latent_delta[:, 1:].mean()),
        "latent_delta_from_zero_max_mean": float(latent_delta[:, 1:].max(axis=1).mean()),
        "latent_candidate_std_mean": float(latent_std.mean()),
        "latent_norm_mean": float(latent_norm.mean()),
        "attention_l1_from_zero_mean": float(attention_l1_from_zero[:, 1:].mean()),
        "attention_l1_from_zero_max_mean": float(
            attention_l1_from_zero[:, 1:].max(axis=1).mean()
        ),
        "attention_normalized_entropy_mean": float(
            (attention_entropy / (max_entropy + 1e-8)).mean()
        ),
        "attention_recency_center_mean": float(recency_center.mean()),
        "predicted_tracking_mean": float(tracking.mean()),
        "predicted_tracking_candidate_std_mean": float(
            jnp.std(jnp.squeeze(tracking, axis=-1), axis=1).mean()
        ),
    }
    if authority.shape[-1] > 0:
        summary.update(
            {
                "predicted_authority_norm_mean": float(
                    jnp.linalg.norm(authority, axis=-1).mean()
                ),
                "predicted_authority_candidate_std_mean": float(
                    jnp.std(authority, axis=1).mean()
                ),
            }
        )
    return summary


def _write_outputs(
    output_dir: Path,
    run_name: str,
    labels: list[str],
    outputs: dict[str, jnp.ndarray],
    summary: dict[str, float],
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    npz_path = output_dir / f"{run_name}_counterfactual_demand_probe.npz"
    csv_path = output_dir / f"{run_name}_counterfactual_demand_summary.csv"

    np.savez_compressed(
        npz_path,
        candidate_labels=np.asarray(labels),
        **{key: np.asarray(value) for key, value in outputs.items()},
    )

    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        for key, value in sorted(summary.items()):
            writer.writerow([key, value])

    print(f"[Counterfactual Probe] saved arrays: {npz_path}")
    print(f"[Counterfactual Probe] saved summary: {csv_path}")


def main() -> None:
    args = _parse_args()
    if args.wandb:
        args.common_args.wandb = True
    else:
        args.common_args.wandb = False

    runner = _load_runner(args)
    data = _load_training_rollout(runner)
    histories, states, policy_actions = _sample_history_windows(
        data,
        history_len=runner.env.history_len,
        max_samples=args.num_samples,
        seed=args.seed,
    )
    labels, candidates = _candidate_wrenches(
        policy_actions,
        runner.env._thruster_mixer_T,
        args.wrench_scale,
    )
    outputs = _probe_module(runner, histories, states, candidates)
    jax.block_until_ready(outputs["mu"])
    summary = _summary_metrics(outputs)

    print("[Counterfactual Probe] summary")
    for key, value in sorted(summary.items()):
        print(f"  {key}: {value:.6f}")

    _write_outputs(args.output_dir, args.run_name, labels, outputs, summary)

    if runner.env.use_wandb:
        import wandb

        wandb.log({f"counterfactual/{k}": v for k, v in summary.items()})
        if wandb.run is not None:
            wandb.finish()


if __name__ == "__main__":
    # Keep the backend setting explicit in logs when launched from train_test_all.sh.
    print(
        "[Counterfactual Probe] SMALLSAT_ROLLOUT_BACKEND="
        f"{os.environ.get('SMALLSAT_ROLLOUT_BACKEND', 'unset')}"
    )
    main()
