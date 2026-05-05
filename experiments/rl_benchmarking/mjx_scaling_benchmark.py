import argparse
import csv
import json
import os
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace

# Keep consistent with other RL benchmarking scripts.
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

# Allow running directly from repository checkout without installation.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np

from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.controllers.rl.runners.rollout import (
    FunctionalRolloutCallbacks,
    prepare_policy_input_with_residuals,
    residuals_from_wrench_delta,
    run_functional_rollout,
)
from smallsat_sim.envs.astrobee_rl.cfg import config as rl_config
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.envs.vec_env import _compute_state_features
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL


@dataclass
class ScalingRow:
    num_envs: int
    steps_per_rollout: int
    sim_dt_per_step: float
    warmup_jit_time_s: float
    first_rollout_time_s: float
    steady_rollout_time_mean_s: float
    steady_rollout_time_std_s: float
    env_steps_per_sec: float
    sim_seconds_per_sec: float
    speedup_vs_min_envs: float
    efficiency_vs_min_envs: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark MJX+JAX nominal training rollout scaling under the same "
            "conditions as ppo_nominal (no failures, no adaptation)."
        )
    )
    parser.add_argument(
        "--num-envs-list",
        type=str,
        default="128,256,512,1024,2048",
        help="Comma-separated env counts to benchmark.",
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--steps-per-rollout",
        type=int,
        default=512,
        help="Control steps in one functional rollout benchmark call.",
    )
    parser.add_argument(
        "--steady-repeats",
        type=int,
        default=1,
        help="Number of post-compile rollout runs for steady-state timing.",
    )
    parser.add_argument(
        "--max-ep-len",
        type=int,
        default=1024,
        help="Episode horizon for rollout reset/truncation logic.",
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/results/mjx_scaling_benchmark",
    )
    return parser.parse_args()


def parse_num_envs(spec: str) -> list[int]:
    values = []
    for token in spec.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    if not values:
        raise ValueError("num-envs-list must contain at least one value.")
    return sorted(set(values))


def make_env_args(headless: bool) -> SimpleNamespace:
    return SimpleNamespace(
        headless=bool(headless),
        num_bodies=1,
        video=False,
        log=False,
        wandb=False,
    )


def run_nominal_rollout(
    runner: OnPolicyRunner,
    *,
    num_steps: int,
    max_ep_len: int,
) -> float:
    env = runner.env
    env.reset()
    env.reset_perturbations()

    step_config = env.build_step_config(max_episode_len=max_ep_len)
    initial_state = env.state_struct
    actor_state, critic_state = runner.agent.actor_critic_state()
    reference = runner.reference_point

    if env.use_adaptive_approach:
        residual_init = jnp.zeros((env.num_envs, env.res_dim), dtype=jnp.float32)
    else:
        residual_init = jnp.zeros((env.num_envs, 0), dtype=jnp.float32)

    def _prepare_policy_input(step_idx, states, residuals, carry_extra):
        return prepare_policy_input_with_residuals(
            step_idx, states, residuals, carry_extra
        )

    def _sample_policy(step_idx, policy_input, rng_key, carry_extra):
        del step_idx  # unused
        rng_key, sample_key = jax.random.split(rng_key)
        actions, values, logp = runner.agent.functional_act(
            actor_state,
            critic_state,
            policy_input,
            sample_key,
        )
        return actions, values, logp, rng_key, carry_extra

    def _post_step(step_idx, step_output, actions, residuals, reset_flag, carry_extra):
        del step_idx, actions, reset_flag  # unused
        residuals_next = residuals_from_wrench_delta(
            step_output=step_output,
            residuals=residuals,
            use_adaptive_approach=env.use_adaptive_approach,
        )
        return residuals_next, None, carry_extra

    def _bootstrap_value(step_idx, env_state, residuals, rng_key, carry_extra):
        rng_key, value_key = jax.random.split(rng_key)
        next_states = _compute_state_features(env_state.mjx_batch, reference)
        policy_input, carry_extra = _prepare_policy_input(
            step_idx, next_states, residuals, carry_extra
        )
        _, values, _ = runner.agent.functional_act(
            actor_state,
            critic_state,
            policy_input,
            value_key,
        )
        return values, rng_key, carry_extra

    t0 = time.perf_counter()
    rollout_result = run_functional_rollout(
        step_config=step_config,
        initial_state=initial_state,
        initial_residuals=residual_init,
        rng=runner.agent.key,
        num_steps=num_steps,
        reference_waypoint=reference,
        callbacks=FunctionalRolloutCallbacks(
            prepare_policy_input=_prepare_policy_input,
            sample_policy=_sample_policy,
            post_step=_post_step,
            bootstrap_value=_bootstrap_value,
        ),
    )
    # Force device synchronization for proper wall-time measurement.
    jax.block_until_ready(rollout_result.actions)
    elapsed = time.perf_counter() - t0

    runner.agent.key = rollout_result.final_rng
    env.apply_state_struct(rollout_result.final_state)
    return elapsed


def write_csv(path: Path, rows: list[ScalingRow]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def plot_results(out_dir: Path, rows: list[ScalingRow]) -> None:
    if not rows:
        return

    num_envs = np.asarray([r.num_envs for r in rows], dtype=float)
    throughput = np.asarray([r.env_steps_per_sec for r in rows], dtype=float)
    sim_rate = np.asarray([r.sim_seconds_per_sec for r in rows], dtype=float)
    speedup = np.asarray([r.speedup_vs_min_envs for r in rows], dtype=float)
    efficiency = np.asarray([r.efficiency_vs_min_envs for r in rows], dtype=float)
    first = np.asarray([r.first_rollout_time_s for r in rows], dtype=float)
    steady = np.asarray([r.steady_rollout_time_mean_s for r in rows], dtype=float)
    warmup = np.asarray([r.warmup_jit_time_s for r in rows], dtype=float)

    plt.figure(figsize=(9, 5))
    plt.plot(num_envs, throughput, marker="o", label="env steps/sec")
    plt.plot(num_envs, sim_rate, marker="s", label="sim seconds/sec")
    plt.xlabel("num envs")
    plt.ylabel("throughput")
    plt.title("MJX Nominal Rollout Throughput Scaling")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "throughput_scaling.png", dpi=180)
    plt.savefig(out_dir / "throughput_scaling.pdf")
    plt.close()

    plt.figure(figsize=(9, 5))
    plt.plot(num_envs, speedup, marker="o", label="speedup vs min envs")
    plt.plot(num_envs, efficiency, marker="s", label="efficiency")
    plt.xlabel("num envs")
    plt.ylabel("relative scaling")
    plt.title("MJX Speedup and Efficiency")
    plt.grid(True, alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "speedup_efficiency.png", dpi=180)
    plt.savefig(out_dir / "speedup_efficiency.pdf")
    plt.close()

    x = np.arange(len(rows))
    labels = [str(int(v)) for v in num_envs]
    width = 0.25
    plt.figure(figsize=(10, 5))
    plt.bar(x - width, warmup, width, label="agent warmup jit")
    plt.bar(x, first, width, label="first rollout (compile+run)")
    plt.bar(x + width, steady, width, label="steady rollout mean")
    plt.xticks(x, labels)
    plt.xlabel("num envs")
    plt.ylabel("time [s]")
    plt.title("MJX Timing Breakdown")
    plt.grid(True, axis="y", alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / "timing_breakdown.png", dpi=180)
    plt.savefig(out_dir / "timing_breakdown.pdf")
    plt.close()


def main() -> None:
    args = parse_args()
    env_counts = parse_num_envs(args.num_envs_list)

    random.seed(args.seed)
    np.random.seed(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[ScalingRow] = []
    baseline_throughput = None

    for num_envs in env_counts:
        print(
            f"[mjx-scale] benchmarking num_envs={num_envs}, "
            f"steps_per_rollout={args.steps_per_rollout}"
        )

        # Apply the env count globally before environment construction.
        rl_config.EnvConfig.control.RL.num_envs = int(num_envs)

        env = AstrobeeEnvVectorized(
            args=make_env_args(args.headless),
            run_name=f"mjx_scale_{num_envs}",
            train_with_failures=False,
            use_pretrained=False,
            use_adaptive_approach=False,
        )
        # Mirror nominal PPO conditions while letting benchmark choose rollout sizes.
        env.env_cfg.control.RL.PPO.steps_per_epoch = int(args.steps_per_rollout)
        env.env_cfg.control.RL.PPO.max_ep_len = int(args.max_ep_len)

        planner = OraclePlannerRL(env, radius=0.0)
        runner = OnPolicyRunner(env, planner)

        t_warm0 = time.perf_counter()
        runner.agent.warmup_jit()
        warmup_jit_time = time.perf_counter() - t_warm0

        first_rollout_time = run_nominal_rollout(
            runner,
            num_steps=int(args.steps_per_rollout),
            max_ep_len=int(args.max_ep_len),
        )

        steady_times = []
        for _ in range(int(args.steady_repeats)):
            t = run_nominal_rollout(
                runner,
                num_steps=int(args.steps_per_rollout),
                max_ep_len=int(args.max_ep_len),
            )
            steady_times.append(float(t))

        steady_mean = float(np.mean(steady_times))
        steady_std = float(np.std(steady_times))
        env_steps = float(num_envs * int(args.steps_per_rollout))
        env_steps_per_sec = env_steps / steady_mean if steady_mean > 0 else float("nan")
        sim_dt_per_step = float(env.env_cfg.sim.dt * env.env_cfg.control.control_decimation)
        sim_seconds_per_sec = env_steps_per_sec * sim_dt_per_step

        if baseline_throughput is None:
            baseline_throughput = env_steps_per_sec
        speedup = env_steps_per_sec / baseline_throughput if baseline_throughput else 1.0
        efficiency = speedup / (num_envs / env_counts[0])

        row = ScalingRow(
            num_envs=int(num_envs),
            steps_per_rollout=int(args.steps_per_rollout),
            sim_dt_per_step=sim_dt_per_step,
            warmup_jit_time_s=float(warmup_jit_time),
            first_rollout_time_s=float(first_rollout_time),
            steady_rollout_time_mean_s=steady_mean,
            steady_rollout_time_std_s=steady_std,
            env_steps_per_sec=float(env_steps_per_sec),
            sim_seconds_per_sec=float(sim_seconds_per_sec),
            speedup_vs_min_envs=float(speedup),
            efficiency_vs_min_envs=float(efficiency),
        )
        rows.append(row)

        print(
            f"[mjx-scale] num_envs={num_envs} "
            f"steady={steady_mean:.3f}s "
            f"throughput={env_steps_per_sec:.1f} steps/s "
            f"speedup={speedup:.2f}x "
            f"eff={efficiency:.2f}"
        )
        env.close()

    csv_path = out_dir / "scaling_metrics.csv"
    json_path = out_dir / "summary.json"
    write_csv(csv_path, rows)
    plot_results(out_dir, rows)

    summary = {
        "config": {
            "seed": int(args.seed),
            "num_envs_list": env_counts,
            "steps_per_rollout": int(args.steps_per_rollout),
            "steady_repeats": int(args.steady_repeats),
            "max_ep_len": int(args.max_ep_len),
            "headless": bool(args.headless),
            "conditions": {
                "train_with_failures": False,
                "use_adaptive_approach": False,
                "benchmark_target": "functional rollout path used in nominal PPO training",
            },
        },
        "rows": [asdict(r) for r in rows],
    }
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("[mjx-scale] complete")
    print(f"[mjx-scale] csv: {csv_path}")
    print(f"[mjx-scale] summary: {json_path}")
    print(f"[mjx-scale] plots: {out_dir}")


if __name__ == "__main__":
    main()
