import argparse
import json
import os
import random
import sys
from pathlib import Path
from types import SimpleNamespace

# Allow running directly from repository checkout without installing as a package.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

import numpy as np
try:
    import wandb  # type: ignore
except ImportError:  # pragma: no cover - optional dependency in this helper
    wandb = None

from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.envs.astrobee_rl.cfg import config as rl_config
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL


# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run many short PPO-nominal trials to quickly compare RL reward/hyperparameter settings."
    )
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument(
        "--sweep-mode",
        type=str,
        choices=("lr_reward_scale10", "random"),
        default="lr_reward_scale10",
        help="Preset 10-point LR/reward-scale grid or random log-uniform sampling.",
    )
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--run-prefix", type=str, default="ppo_nominal_quick")
    parser.add_argument("--num-envs", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--steps-per-epoch", type=int, default=256)
    parser.add_argument("--max-ep-len", type=int, default=256)
    parser.add_argument("--episode-len", type=int, default=256)
    parser.add_argument("--n-evals", type=int, default=2)
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--video",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--wandb",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/rl_results/quick_tune_ppo_nominal",
    )
    return parser.parse_args()


def _build_trial_params(
    *,
    reward_scale: float,
    actor_lr: float,
    critic_lr: float,
    design_point: str,
) -> dict:
    return {
        # PPO learning hyperparameters
        "actor_lr": float(actor_lr),
        "critic_lr": float(critic_lr),
        "entropy_coef": 1e-4,
        "clip_ratio": 0.2,
        "gamma": 0.99,
        "lam": 0.95,
        # Reward weights (scaled together to preserve objective shape)
        "w_pos": float(5e1 * reward_scale),
        "w_vel": float(1e1 * reward_scale),
        "w_att": float(2e1 * reward_scale),
        "w_angvel": 0.0,
        # Penalty weights (scaled together with rewards)
        "lam_fuel": float(1e-4 * reward_scale),
        "lam_speed_terminal": float(1e-1 * reward_scale),
        "lam_ang_speed_terminal": float(1e-3 * reward_scale),
        "lam_fuel_terminal": float(5e-4 * reward_scale),
        "terminal_bonus": float(1e0 * reward_scale),
        # Metadata for analysis / logging
        "reward_scale": float(reward_scale),
        "design_point": design_point,
    }


def _build_lr_reward_scale10_plan() -> list[dict]:
    reward_scales = (0.5, 1.0, 2.0)
    lr_pairs = (
        (1.5e-4, 3e-4),
        (3e-4, 6e-4),
        (6e-4, 1e-3),
    )

    plan: list[dict] = []
    for reward_scale in reward_scales:
        for actor_lr, critic_lr in lr_pairs:
            design_point = f"s{reward_scale:g}_alr{actor_lr:.1e}_clr{critic_lr:.1e}"
            plan.append(
                _build_trial_params(
                    reward_scale=reward_scale,
                    actor_lr=actor_lr,
                    critic_lr=critic_lr,
                    design_point=design_point,
                )
            )

    # 10th run: center point repeat for variance check.
    plan.append(
        _build_trial_params(
            reward_scale=1.0,
            actor_lr=3e-4,
            critic_lr=6e-4,
            design_point="center_repeat",
        )
    )
    return plan


def _sample_log_uniform(rng: random.Random, lo: float, hi: float) -> float:
    return float(np.exp(rng.uniform(np.log(lo), np.log(hi))))


def _sample_trial_params(rng: random.Random) -> dict:
    params = {
        # PPO learning hyperparameters
        "actor_lr": _sample_log_uniform(rng, 1e-4, 5e-4),
        "critic_lr": _sample_log_uniform(rng, 2e-4, 1e-3),
        "entropy_coef": _sample_log_uniform(rng, 1e-6, 5e-4),
        "clip_ratio": float(rng.uniform(0.12, 0.28)),
        "gamma": float(rng.uniform(0.985, 0.999)),
        "lam": float(rng.uniform(0.90, 0.98)),
        # Reward weights
        "w_pos": _sample_log_uniform(rng, 2e1, 2e2),
        "w_vel": _sample_log_uniform(rng, 1e0, 5e1),
        "w_att": _sample_log_uniform(rng, 5e0, 8e1),
        "w_angvel": _sample_log_uniform(rng, 1e-2, 1e1),
        # Penalty weights
        "lam_fuel": _sample_log_uniform(rng, 1e-5, 5e-3),
        "lam_speed_terminal": _sample_log_uniform(rng, 1e-2, 5e-1),
        "lam_ang_speed_terminal": _sample_log_uniform(rng, 1e-4, 5e-2),
        "lam_fuel_terminal": _sample_log_uniform(rng, 1e-5, 5e-3),
        "terminal_bonus": _sample_log_uniform(rng, 5e-1, 2e1),
    }
    # Keep critic at least as fast as actor in this quick screen.
    params["critic_lr"] = max(params["critic_lr"], params["actor_lr"])
    return params


def _apply_trial_overrides(
    env: AstrobeeEnvVectorized,
    *,
    trial_params: dict,
    epochs: int,
    steps_per_epoch: int,
    max_ep_len: int,
    episode_len: int,
    n_evals: int,
) -> None:
    rl = env.env_cfg.control.RL
    ppo = rl.PPO

    ppo.epochs = int(epochs)
    ppo.steps_per_epoch = int(steps_per_epoch)
    ppo.max_ep_len = int(max_ep_len)
    ppo.actor_lr = float(trial_params["actor_lr"])
    ppo.critic_lr = float(trial_params["critic_lr"])
    ppo.entropy_coef = float(trial_params["entropy_coef"])
    ppo.clip_ratio = float(trial_params["clip_ratio"])
    ppo.gamma = float(trial_params["gamma"])
    ppo.lam = float(trial_params["lam"])

    rl.curriculum_nominal_epochs = int(epochs)
    rl.curriculum_phase_epochs = max(1, int(epochs))
    rl.episode_len = int(episode_len)
    rl.n_evals = int(n_evals)

    rl.w_pos = float(trial_params["w_pos"])
    rl.w_vel = float(trial_params["w_vel"])
    rl.w_att = float(trial_params["w_att"])
    rl.w_angvel = float(trial_params["w_angvel"])

    rl.lam_fuel = float(trial_params["lam_fuel"])
    rl.lam_speed_terminal = float(trial_params["lam_speed_terminal"])
    rl.lam_ang_speed_terminal = float(trial_params["lam_ang_speed_terminal"])
    rl.lam_fuel_terminal = float(trial_params["lam_fuel_terminal"])
    rl.terminal_bonus = float(trial_params["terminal_bonus"])

    # Mirror into VecEnv cached scalars.
    env._load_vec_env_hyperparams()


def _make_args(ns: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        headless=bool(ns.headless),
        num_bodies=1,
        video=bool(ns.video),
        log=bool(ns.log),
        wandb=bool(ns.wandb),
    )


def main() -> None:
    args = _parse_args()
    if args.wandb and wandb is None:
        raise RuntimeError(
            "Weights & Biases is not installed but --wandb is enabled. "
            "Install wandb or run with --no-wandb."
        )
    rng = random.Random(args.seed)
    np.random.seed(args.seed)

    out_root = Path(args.out_dir).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # Global config override before env construction (batch shape depends on num_envs).
    rl_config.EnvConfig.control.RL.num_envs = int(args.num_envs)

    if args.sweep_mode == "lr_reward_scale10":
        trial_plan = _build_lr_reward_scale10_plan()
        if args.trials < len(trial_plan):
            trial_plan = trial_plan[: args.trials]
        elif args.trials > len(trial_plan):
            print(
                f"[Info] Requested trials={args.trials} but preset has {len(trial_plan)} points; "
                f"running preset size only."
            )
    else:
        trial_plan = [_sample_trial_params(rng) for _ in range(args.trials)]

    print(
        f"Launching quick nominal PPO tuning: sweep_mode={args.sweep_mode}, "
        f"trials={len(trial_plan)}, num_envs={args.num_envs}, "
        f"epochs={args.epochs}, steps_per_epoch={args.steps_per_epoch}"
    )

    for trial_idx, trial_params in enumerate(trial_plan):
        trial_name = f"{args.run_prefix}_t{trial_idx:03d}"
        trial_dir = out_root / trial_name
        ckpt_dir = trial_dir / "checkpoints"
        trial_dir.mkdir(parents=True, exist_ok=True)
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        (trial_dir / "trial_params.json").write_text(
            json.dumps(trial_params, indent=2),
            encoding="utf-8",
        )

        print(f"\n[{trial_idx + 1}/{len(trial_plan)}] {trial_name}")
        print(json.dumps(trial_params, indent=2))

        env = AstrobeeEnvVectorized(
            args=_make_args(args),
            run_name=trial_name,
            train_with_failures=False,
            use_pretrained=False,
            use_adaptive_approach=False,
        )
        _apply_trial_overrides(
            env,
            trial_params=trial_params,
            epochs=args.epochs,
            steps_per_epoch=args.steps_per_epoch,
            max_ep_len=args.max_ep_len,
            episode_len=args.episode_len,
            n_evals=args.n_evals,
        )

        planner = OraclePlannerRL(env, radius=0.0)
        runner = OnPolicyRunner(env, planner)
        runner.ckpt_dir = str(ckpt_dir) + "/"

        if env.use_wandb and wandb is not None and wandb.run is not None:
            wandb.config.update(
                {
                    "sweep_type": f"quick_nominal_screen_{args.sweep_mode}",
                    "trial_idx": trial_idx,
                    **trial_params,
                    "quick_epochs": args.epochs,
                    "quick_steps_per_epoch": args.steps_per_epoch,
                    "quick_max_ep_len": args.max_ep_len,
                    "quick_episode_len": args.episode_len,
                    "quick_n_evals": args.n_evals,
                },
                allow_val_change=True,
            )

        try:
            runner.learn()
            runner.evaluate(phase=1)
            if args.log:
                env.logger.save_log()
        finally:
            if env.use_wandb and wandb is not None and wandb.run is not None:
                wandb.finish()


if __name__ == "__main__":
    main()
