import os
import time

import numpy as np
import jax.numpy as jnp

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

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
    if not getattr(args, "video", False):
        args.video = True
    if not getattr(args, "headless", False):
        args.headless = True

    run_name = "pd_training_env"
    output_root = _ensure_dir(
        os.path.join(
            "experiments",
            "rl_results",
            run_name,
            time.strftime("%Y%m%d_%H%M%S"),
        )
    )

    # Training env, same config as RL
    env = AstrobeeEnvVectorized(
        args,
        run_name=run_name,
        train_with_failures=True,
    )

    # Use the same oracle planner as training (setpoint when radius=0.0)
    planner = OraclePlannerRL(env, radius=0.0)
    pd_ctrl = VectorizedPDController(env, planner)

    dt = env.env_cfg.sim.dt * env.env_cfg.control.control_decimation
    max_steps = int(env.env_cfg.control.RL.max_ep_len)

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
                env._update_renderer()

        # Update residuals if needed
        if env.use_adaptive_approach:
            actual_wrench = env.get_actual_wrench()
            desired_wrench = env.get_desired_wrench(actions)
            residuals = actual_wrench - desired_wrench

        pos = np.asarray(env.mjx_batch.qpos[0, :3])
        ref_pos = np.asarray(reference[0, :3])
        quat = np.asarray(obs[0, 3:7])
        ref_quat = np.asarray(reference[0, 3:7])

        pos_traj.append(pos)
        ref_traj.append(ref_pos)
        pos_err.append(np.linalg.norm(pos - ref_pos))
        att_err.append(float(np.degrees(calc_attitude_error(ref_quat, quat))))
        rewards.append(float(r[0]))

        if bool(terminal[0]):
            break

    pos_traj = np.asarray(pos_traj)
    ref_traj = np.asarray(ref_traj)
    rewards = np.asarray(rewards)
    pos_err = np.asarray(pos_err)
    att_err = np.asarray(att_err)
    times = np.arange(len(rewards)) * dt
    cumulative_return = np.cumsum(rewards)

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
