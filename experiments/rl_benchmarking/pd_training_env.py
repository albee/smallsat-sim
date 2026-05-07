import os
import time

# Ensure a headless-safe GL backend is selected before MuJoCo imports.
if "MUJOCO_GL" not in os.environ:
    os.environ["MUJOCO_GL"] = "osmesa"

import numpy as np
import jax.numpy as jnp
from mujoco import mjx

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

from smallsat_sim.utils.helpers import get_args, calc_attitude_error
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.controllers.pd.vectorized_controller import VectorizedPDController


def _ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def run_pd_benchmark() -> None:
    args = get_args()

    # Force video + headless unless explicitly disabled via CLI
    if not getattr(args, "headless", False):
        args.headless = True


    run_name = "pd_training_env"
    repo_root = Path(__file__).resolve().parents[2]
    output_root = _ensure_dir(
        os.path.join(
            str(repo_root),
            "experiments",
            "rl_results",
            run_name,
            time.strftime("%Y%m%d_%H%M%S"),
        )
    )

    # Training env (MJX) to match pd_controller conditions
    env = AstrobeeEnvVectorized(
        args,
        run_name=run_name,
        train_with_failures=True,
    )

    # Use the same oracle planner as training (setpoint when radius=0.0)
    planner = OraclePlannerRL(env, radius=0.0)
    pd_ctrl = VectorizedPDController(env, planner)

    dt = env.env_cfg.sim.dt * env.env_cfg.control.control_decimation
    max_steps = int(
        getattr(env.env_cfg.control.RL, "max_ep_len", None)
        or env.env_cfg.control.RL.PPO.max_ep_len
    )
    print(f"[pd_training_env] max_steps={max_steps}")

    # Initialize rollout
    env.reset()
    if env.use_adaptive_approach:
        residuals = jnp.zeros((env.num_envs, env.res_dim), dtype=jnp.float32)
    else:
        residuals = jnp.zeros((env.num_envs, 0), dtype=jnp.float32)

    pos_traj = []
    ref_traj = []
    pos_err = []
    att_err = []
    rewards = []
    saved_preview = False

    for step in range(max_steps):
        obs = env.get_obs()
        reference = planner.get_reference(obs)

        actions = pd_ctrl.get_control_input(env, next_waypoint=reference)
        states = env.get_states(reference)
        state_res = (
            jnp.concatenate([states, residuals], axis=1)
            if residuals.shape[1] > 0
            else states
        )

        r, terminal = env.transition(actions, state_res, reference)

        # Capture video frames during the recording window
        if args.video:
            if (
                env.data.time >= env.env_cfg.renderer.start_recording
                and env.data.time <= env.env_cfg.renderer.end_recording
            ):                
                # Visualize reference points (same path as pd_controller)
                if hasattr(planner, "reference_point_list"):
                    env._visualize_renderer(planner.reference_point_list)
                mjx.get_data_into(env.data_vec, env.model, env.mjx_batch)
                env._update_renderer()
                if not saved_preview and env.frames:
                    plt.imsave(
                        os.path.join(output_root, "frame0.png"), env.frames[-1]
                    )
                    saved_preview = True

        # Update residuals if needed
        if env.use_adaptive_approach:
            actual_wrench = env.get_actual_wrench()
            desired_wrench = env.get_desired_wrench(actions)
            residuals = actual_wrench - desired_wrench

        pos = np.asarray(env.mjx_batch.qpos[0, :3])
        ref_pos_plot = np.asarray(reference[0, :3])
        quat = np.asarray(obs[0, 3:7])
        ref_quat_plot = np.asarray(reference[0, 3:7])

        pos_traj.append(pos)
        ref_traj.append(ref_pos_plot)
        pos_err.append(np.linalg.norm(pos - ref_pos_plot))
        att_err.append(float(calc_attitude_error(ref_quat_plot, quat)))
        rewards.append(float(r[0]))

    pos_traj = np.asarray(pos_traj)
    ref_traj = np.asarray(ref_traj)
    rewards = np.asarray(rewards)
    pos_err = np.asarray(pos_err)
    att_err = np.asarray(att_err)
    times = np.arange(len(rewards)) * dt
    cumulative_return = np.cumsum(rewards)
    print(
        f"[pd_training_env] samples={len(rewards)} "
        f"pos_err_nan={np.isnan(pos_err).any()} att_err_nan={np.isnan(att_err).any()} reward_nan={np.isnan(rewards).any()}"
    )
    if len(rewards) > 0:
        print(
            f"[pd_training_env] pos_err(first,last)=({pos_err[0]:.4f},{pos_err[-1]:.4f}) "
            f"att_err(first,last)=({att_err[0]:.4f},{att_err[-1]:.4f}) "
            f"reward(first,last)=({rewards[0]:.4f},{rewards[-1]:.4f})"
        )

    # Plot: trajectory vs reference
    fig = plt.figure(figsize=(9, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        pos_traj[:, 0],
        pos_traj[:, 1],
        pos_traj[:, 2],
        label="PD trajectory",
        linewidth=2,
    )
    if ref_traj.shape[0] > 1:
        ax.plot(
            ref_traj[:, 0],
            ref_traj[:, 1],
            ref_traj[:, 2],
            label="Reference",
            linestyle="--",
            linewidth=1.5,
        )
    else:
        ax.scatter(
            ref_traj[:, 0],
            ref_traj[:, 1],
            ref_traj[:, 2],
            label="Reference",
            s=60,
        )
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")
    ax.set_title("PD Trajectory vs Reference")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(output_root, "trajectory_3d.png"), dpi=200)
    plt.close(fig)

    # Plot: errors and returns
    fig, axs = plt.subplots(3, 1, figsize=(10, 10), sharex=True)
    axs[0].plot(times, pos_err, label="Position error")
    axs[0].set_ylabel("Position error (m)")
    axs[0].grid(True, alpha=0.3)

    axs[1].plot(times, att_err, label="Attitude error", color="tab:orange")
    axs[1].set_ylabel("Attitude error (deg)")
    axs[1].grid(True, alpha=0.3)

    axs[2].plot(times, rewards, label="Reward", color="tab:green")
    axs[2].plot(times, cumulative_return, label="Cumulative return", color="tab:purple")
    axs[2].set_xlabel("Time (s)")
    axs[2].set_ylabel("Reward")
    axs[2].grid(True, alpha=0.3)
    axs[2].legend()

    fig.suptitle("PD Tracking Errors and Returns")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(os.path.join(output_root, "errors_and_returns.png"), dpi=200)
    plt.close(fig)

    # Save video
    if args.video:
        env.get_sim_rendering("pd_training_env", output_dir=output_root)

    print(f"PD benchmark complete. Outputs saved to: {output_root}")
    print(f"Total return: {float(cumulative_return[-1]) if cumulative_return.size else 0.0}")


if __name__ == "__main__":
    run_pd_benchmark()
