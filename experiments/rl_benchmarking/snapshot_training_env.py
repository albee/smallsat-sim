import argparse
import os
import sys
from pathlib import Path
from types import SimpleNamespace

# Configure backends before importing JAX / MuJoCo wrappers.
os.environ["MUJOCO_GL"] = "osmesa"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
from mujoco import mjx

# Allow running directly from repository checkout without package install.
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from smallsat_sim.envs.astrobee_rl.cfg import config as rl_config
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Initialize the RL training environment on CPU and save a renderer snapshot."
    )
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--num-envs",
        type=int,
        default=1,
        help="Number of vectorized environments (use 1 for CPU snapshot).",
    )
    parser.add_argument(
        "--steps",
        type=int,
        default=0,
        help="Optional number of zero-action transitions before capturing.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="experiments/rl_results/rl_training_env_snapshot/training_env_snapshot.png",
        help="Path to save the PNG snapshot.",
    )
    parser.add_argument(
        "--train-with-failures",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Enable RL training failures while building the environment.",
    )
    parser.add_argument(
        "--setpoint-radius",
        type=float,
        default=0.0,
        help=(
            "Radius [m] of OraclePlannerRL setpoints. "
            "Use 0.0 for training-like single-setpoint regulation."
        ),
    )
    parser.add_argument(
        "--setpoint-spacing",
        type=float,
        default=1.0,
        help="Spacing [m] between OraclePlannerRL setpoints.",
    )
    parser.add_argument(
        "--max-panels",
        type=int,
        default=9,
        help="Maximum number of env panels to render in the grid snapshot.",
    )
    parser.add_argument(
        "--grid-cols",
        type=int,
        default=3,
        help="Number of columns in the panel grid.",
    )
    parser.add_argument(
        "--panel-padding",
        type=float,
        default=4.0,
        help="Pixel padding between panels.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=("grid", "init_cloud", "trajectory_fan"),
        default="grid",
        help=(
            "Snapshot mode: 'grid' renders multiple env panels; "
            "'init_cloud' renders env0 plus N-1 oriented grey cubes at other env initial states; "
            "'trajectory_fan' plots short trajectory traces from many parallel envs."
        ),
    )
    parser.add_argument(
        "--cloud-max-markers",
        type=int,
        default=128,
        help="Maximum number of grey cloud markers to draw in init_cloud mode.",
    )
    parser.add_argument(
        "--cloud-marker-size",
        type=float,
        default=0.08,
        help="Half-size [m] of grey cube markers in init_cloud mode.",
    )
    parser.add_argument(
        "--cloud-alpha",
        type=float,
        default=0.2,
        help="Alpha transparency of grey cloud cubes in init_cloud mode.",
    )
    parser.add_argument(
        "--lookat-z-offset",
        type=float,
        default=0.35,
        help=(
            "Screen-framing offset [m]. Positive values move setpoints upward "
            "in the final image."
        ),
    )
    parser.add_argument(
        "--fan-steps",
        type=int,
        default=120,
        help="Number of rollout steps to collect for trajectory_fan mode.",
    )
    parser.add_argument(
        "--fan-max-traces",
        type=int,
        default=128,
        help="Maximum number of env trajectories to draw in trajectory_fan mode.",
    )
    return parser.parse_args()


def _render_env_with_setpoints(
    env: AstrobeeEnvVectorized,
    env_index: int,
    setpoints: jnp.ndarray,
) -> np.ndarray:
    """
    Render one environment frame with planner setpoints overlaid.
    """
    env.renderer.update_scene(env.data_vec[int(env_index)], env.cam)
    scene = env.renderer.scene
    max_extra = max(0, int(scene.maxgeom) - int(scene.ngeom))

    pts = np.asarray(setpoints, dtype=float).reshape(-1, 3)
    n_pts = max(0, min(int(pts.shape[0]), max_extra))
    for i in range(n_pts):
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.05, 0.0, 0.0]),
            pos=np.asarray(pts[i], dtype=float),
            mat=np.eye(3).reshape(-1),
            rgba=np.array([0.1, 0.8, 1.0, 0.95]),
        )

    return env.renderer.render().copy()


def _lock_grid_camera(
    env: AstrobeeEnvVectorized,
    setpoints: jnp.ndarray,
    *,
    mode: str = "grid",
    lookat_z_offset: float = 0.0,
) -> None:
    """
    Force a fixed camera pose for all panel renders (no body tracking).
    """
    # Snapshot current framing from env 0 first.
    env.renderer.update_scene(env.data_vec[0], env.cam)

    # Convert to free camera so it does not re-center per env instance.
    env.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    env.cam.trackbodyid = -1

    pts = np.asarray(setpoints, dtype=float).reshape(-1, 3)
    if pts.shape[0] > 0:
        # Keep setpoint(s) centered in the final screenshot.
        target_center = np.mean(pts, axis=0)
        if mode == "init_cloud" and int(env.num_envs) > 1:
            # For init cloud, bias center toward cloud centroid for better framing.
            cloud_positions = np.asarray(env.mjx_batch.qpos[:, :3], dtype=float)
            cloud_center = np.mean(cloud_positions, axis=0)
            target_center = 0.55 * target_center + 0.45 * cloud_center
        # Positive value should move the setpoint upward in image space.
        target_center[2] -= float(lookat_z_offset)
        env.cam.lookat[:] = target_center
    else:
        gateway_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gateway_full")
        if gateway_id >= 0:
            env.cam.lookat[:] = np.asarray(env.data_vec[0].xpos[gateway_id], dtype=float)


def _make_panel_grid(
    frames: list[np.ndarray],
    *,
    cols: int,
    padding_px: int,
) -> np.ndarray:
    if not frames:
        raise ValueError("No frames provided for panel grid.")
    if len(frames) == 1:
        return frames[0]

    h, w, c = frames[0].shape
    cols = max(1, int(cols))
    rows = int(np.ceil(len(frames) / cols))
    pad = max(0, int(padding_px))

    canvas_h = rows * h + max(0, rows - 1) * pad
    canvas_w = cols * w + max(0, cols - 1) * pad
    canvas = np.zeros((canvas_h, canvas_w, c), dtype=np.uint8)

    for i, frame in enumerate(frames):
        r = i // cols
        cidx = i % cols
        y0 = r * (h + pad)
        x0 = cidx * (w + pad)
        canvas[y0 : y0 + h, x0 : x0 + w] = frame

    return canvas


def _plot_trajectory_fan(
    output_path: Path,
    positions_xyz: np.ndarray,
    setpoints: np.ndarray,
    *,
    max_traces: int,
) -> None:
    """
    Plot trajectory fan in x-y with one line per parallel env.
    """
    # positions_xyz: [T, N, 3]
    t_steps, n_envs, _ = positions_xyz.shape
    if t_steps < 2 or n_envs < 1:
        raise ValueError("Need at least 2 timesteps and 1 env for trajectory fan.")

    max_traces = max(1, int(max_traces))
    if n_envs <= max_traces:
        env_indices = np.arange(n_envs, dtype=int)
    else:
        env_indices = np.linspace(0, n_envs - 1, num=max_traces, dtype=int)

    fig, ax = plt.subplots(figsize=(7.2, 7.2))
    for idx in env_indices:
        xy = positions_xyz[:, idx, :2]
        ax.plot(xy[:, 0], xy[:, 1], color="#1F4E79", alpha=0.14, lw=1.1)

    # Highlight env 0 trajectory.
    xy0 = positions_xyz[:, 0, :2]
    ax.plot(xy0[:, 0], xy0[:, 1], color="#0B1F3A", alpha=0.95, lw=2.0)

    # Plot setpoint(s).
    sp = np.asarray(setpoints, dtype=float).reshape(-1, 3)
    ax.scatter(sp[:, 0], sp[:, 1], s=30, c="#2AB7CA", edgecolors="white", linewidths=0.4)

    ax.set_xlabel("x [m]")
    ax.set_ylabel("y [m]")
    ax.grid(True, alpha=0.18)
    ax.set_aspect("equal", adjustable="box")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    fig.tight_layout()
    fig.savefig(output_path, dpi=260)
    fig.savefig(output_path.with_suffix(".pdf"))
    plt.close(fig)


def _render_init_cloud(
    env: AstrobeeEnvVectorized,
    setpoints: jnp.ndarray,
    *,
    max_markers: int,
    marker_size: float,
    cloud_alpha: float,
) -> np.ndarray:
    """
    Render env0 with setpoints, plus oriented grey cubes at other env states.
    """
    env.renderer.update_scene(env.data_vec[0], env.cam)
    scene = env.renderer.scene

    # Setpoint overlays.
    pts = np.asarray(setpoints, dtype=float).reshape(-1, 3)
    max_extra = max(0, int(scene.maxgeom) - int(scene.ngeom))
    n_pts = max(0, min(int(pts.shape[0]), max_extra))
    for i in range(n_pts):
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_SPHERE,
            size=np.array([0.05, 0.0, 0.0]),
            pos=np.asarray(pts[i], dtype=float),
            mat=np.eye(3).reshape(-1),
            rgba=np.array([0.1, 0.8, 1.0, 0.95]),
        )

    # N-1 env markers as oriented grey cubes.
    n_envs = int(env.num_envs)
    remaining_geom = max(0, int(scene.maxgeom) - int(scene.ngeom))
    n_markers = min(max(0, n_envs - 1), int(max_markers), remaining_geom)
    size_vec = np.array([marker_size, marker_size, marker_size], dtype=float)
    rgba = np.array(
        [0.65, 0.65, 0.65, float(np.clip(cloud_alpha, 0.02, 1.0))],
        dtype=float,
    )
    rot_mat = np.zeros(9, dtype=float)

    for env_idx in range(1, 1 + n_markers):
        qpos = np.asarray(env.data_vec[env_idx].qpos, dtype=float)
        pos = qpos[:3]
        quat = qpos[3:7]  # MuJoCo quaternion format [w, x, y, z]
        mujoco.mju_quat2Mat(rot_mat, quat)
        scene.ngeom += 1
        mujoco.mjv_initGeom(
            scene.geoms[scene.ngeom - 1],
            type=mujoco.mjtGeom.mjGEOM_BOX,
            size=size_vec,
            pos=pos,
            mat=rot_mat,
            rgba=rgba,
        )

    return env.renderer.render().copy()


def _ensure_renderer_capacity(env: AstrobeeEnvVectorized, required_maxgeom: int) -> None:
    """
    Recreate renderer with larger max_geom budget when many overlay geoms are needed.
    """
    if env.renderer is None:
        return

    current = int(getattr(env.renderer.scene, "maxgeom", 0))
    target = max(int(required_maxgeom), current)
    if current >= target:
        return

    width = int(env.env_cfg.renderer.width)
    height = int(env.env_cfg.renderer.height)
    try:
        new_renderer = mujoco.Renderer(
            env.model,
            width=width,
            height=height,
            max_geom=target,
        )
    except TypeError:
        # Older mujoco python bindings may not expose max_geom in Renderer ctor.
        print(
            "[snapshot] warning: renderer max_geom not configurable in this MuJoCo build; "
            f"current maxgeom={current}"
        )
        return

    old_renderer = env.renderer
    env.renderer = new_renderer
    try:
        old_renderer.close()
    except Exception:
        pass


def main() -> None:
    args = parse_args()

    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    rl_config.EnvConfig.control.RL.num_envs = int(args.num_envs)

    env_args = SimpleNamespace(
        headless=True,
        num_bodies=1,
        video=True,
        log=False,
        wandb=False,
    )

    env = None
    try:
        env = AstrobeeEnvVectorized(
            env_args,
            run_name="rl_training_env_snapshot",
            train_with_failures=bool(args.train_with_failures),
            use_pretrained=False,
            use_adaptive_approach=False,
        )
        if not hasattr(env.env_cfg.control, "control_decimation"):
            env.env_cfg.control.control_decimation = (
                env.env_cfg.control.RL.control_decimation
            )

        # Deterministic reset for reproducible snapshots.
        env._rng = jax.random.PRNGKey(int(args.seed))
        env.reset()
        planner = OraclePlannerRL(
            env,
            radius=float(args.setpoint_radius),
            spacing=float(args.setpoint_spacing),
        )

        if args.mode == "trajectory_fan":
            fan_steps = max(2, int(args.fan_steps))
            pos_seq = []
            # Log initial batch positions.
            pos_seq.append(np.asarray(env.mjx_batch.qpos[:, :3], dtype=float).copy())
            for _ in range(fan_steps):
                ctrl = jnp.zeros((env.num_envs, env.act_dim), dtype=jnp.float32)
                obs = env.get_obs()
                ref = planner.get_reference(obs)
                state = env.get_states(ref)
                env.transition(ctrl, state, ref)
                pos_seq.append(np.asarray(env.mjx_batch.qpos[:, :3], dtype=float).copy())

            positions_xyz = np.stack(pos_seq, axis=0)
            _plot_trajectory_fan(
                output_path=output_path,
                positions_xyz=positions_xyz,
                setpoints=np.asarray(planner.reference_points, dtype=float),
                max_traces=int(args.fan_max_traces),
            )

            print(f"[snapshot] JAX backend: {jax.default_backend()}")
            print(f"[snapshot] num_envs={env.num_envs}, fan_steps={fan_steps}")
            print(f"[snapshot] trajectory_fan: max_traces={int(args.fan_max_traces)}")
            print(f"[snapshot] saved: {output_path}")
            return

        if args.steps > 0 and args.mode != "init_cloud":
            for _ in range(int(args.steps)):
                ctrl = jnp.zeros((env.num_envs, env.act_dim), dtype=jnp.float32)
                obs = env.get_obs()
                ref = planner.get_reference(obs)
                state = env.get_states(ref)
                env.transition(ctrl, state, ref)

        mjx.get_data_into(env.data_vec, env.model, env.mjx_batch)
        if args.mode == "init_cloud":
            requested_markers = min(
                max(0, int(env.num_envs) - 1),
                max(0, int(args.cloud_max_markers)),
            )
            # Reserve extra room for station geoms + setpoint overlays.
            _ensure_renderer_capacity(
                env,
                required_maxgeom=int(requested_markers + 2000),
            )
        _lock_grid_camera(
            env,
            planner.reference_points,
            mode=str(args.mode),
            lookat_z_offset=float(args.lookat_z_offset),
        )
        if args.mode == "grid":
            max_panels = max(1, int(args.max_panels))
            num_panels = min(int(env.num_envs), max_panels)
            frames = [
                _render_env_with_setpoints(
                    env,
                    env_index=i,
                    setpoints=planner.reference_points,
                )
                for i in range(num_panels)
            ]
            frame = _make_panel_grid(
                frames,
                cols=int(args.grid_cols),
                padding_px=int(args.panel_padding),
            )
        else:
            frame = _render_init_cloud(
                env,
                planner.reference_points,
                max_markers=int(args.cloud_max_markers),
                marker_size=float(args.cloud_marker_size),
                cloud_alpha=float(args.cloud_alpha),
            )
        plt.imsave(output_path, frame)

        print(f"[snapshot] JAX backend: {jax.default_backend()}")
        print(f"[snapshot] num_envs={env.num_envs}, steps={int(args.steps)}")
        print(
            "[snapshot] setpoints: "
            f"radius={float(args.setpoint_radius):.2f}, spacing={float(args.setpoint_spacing):.2f}, "
            f"count={int(planner.reference_points.shape[0])}"
        )
        if args.mode == "grid":
            print(f"[snapshot] panel grid: panels={num_panels}, cols={int(args.grid_cols)}")
        else:
            print(
                "[snapshot] init_cloud: "
                f"markers={min(max(0, int(env.num_envs) - 1), int(args.cloud_max_markers))}, "
                f"marker_size={float(args.cloud_marker_size):.3f}, "
                f"cloud_alpha={float(args.cloud_alpha):.2f}"
            )
        print(f"[snapshot] saved: {output_path}")
    finally:
        if env is not None:
            env.close()


if __name__ == "__main__":
    main()
