import argparse
import csv
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import matplotlib.pyplot as plt
import mujoco
import numpy as np

from smallsat_sim.controllers.nominal_mpc.controller import NominalMPCController
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.envs.astrobee.cfg.config import randomize_initial_state
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.planners.mission.mission import MissionPlanner
from smallsat_sim.utils.helpers import calc_lateral_tracking_error


@dataclass
class TrialScenario:
    trial_id: int
    init_pos: list[float]
    init_att: list[float]
    failure_time: float | None
    failure_thrusters: list[int] | None


@dataclass
class TrialMetrics:
    controller: str
    trial_id: int
    success: int
    progress_fraction: float
    mean_lateral_error: float
    rms_lateral_error: float
    control_effort: float
    mean_distance_to_waypoint: float
    final_distance_to_waypoint: float
    sim_time: float


@dataclass
class TrialTrace:
    controller: str
    trial_id: int
    time: list[float]
    x: list[float]
    y: list[float]
    z: list[float]
    distance_to_waypoint: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Inspection benchmark for PD vs NominalMPC with optional per-thruster "
            "stuck-on/off failures."
        )
    )
    parser.add_argument(
        "--controllers",
        type=str,
        default="pd,nominal_mpc",
        help="Comma-separated list from {pd, nominal_mpc}.",
    )
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--max-sim-time", type=float, default=30.0)
    parser.add_argument(
        "--planner-spacing",
        type=float,
        default=0.4,
        help="Spacing [m] for intermediate mission references.",
    )
    parser.add_argument(
        "--planner-clearance-dist",
        type=float,
        default=0.35,
        help="Distance [m] to mark an intermediate reference as reached.",
    )
    parser.add_argument(
        "--failure-mode",
        choices=("nominal", "stuck_off", "stuck_on"),
        default="nominal",
    )
    parser.add_argument(
        "--failure-time",
        type=float,
        default=1.0,
        help="Sim-time [s] when stuck failure is injected.",
    )
    parser.add_argument(
        "--failure-thruster",
        type=int,
        default=None,
        help=(
            "Optional fixed primary thruster index. "
            "If omitted, failed thrusters are sampled each trial."
        ),
    )
    parser.add_argument(
        "--failure-count",
        type=int,
        default=1,
        help=(
            "Number of thrusters to fail per trial for stuck_off/stuck_on modes. "
            "Defaults to 1."
        ),
    )
    parser.add_argument(
        "--success-progress",
        type=float,
        default=0.95,
        help="Required mission progress fraction to count as success.",
    )
    parser.add_argument(
        "--pd-kp-x",
        type=float,
        default=0.8,
        help="Optional PD translational proportional gain override.",
    )
    parser.add_argument(
        "--pd-kd-x",
        type=float,
        default=2.0,
        help="Optional PD translational derivative gain override.",
    )
    parser.add_argument(
        "--pd-kp-q",
        type=float,
        default=6.0,
        help="Optional PD attitude proportional gain override.",
    )
    parser.add_argument(
        "--pd-kd-q",
        type=float,
        default=8.0,
        help="Optional PD attitude derivative gain override.",
    )
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
        "--save-snapshot",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Save deterministic multi-view snapshots at t=5s for each controller/trial. "
            "Each frame includes station, smallsat, and trajectory-so-far."
        ),
    )
    parser.add_argument(
        "--snapshot-time",
        type=float,
        default=5.0,
        help="Sim-time [s] at which multi-view snapshots are captured.",
    )
    parser.add_argument(
        "--snapshot-views",
        type=int,
        default=10,
        help="Number of camera viewpoints captured at snapshot time.",
    )
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/results/inspection_benchmark",
    )
    return parser.parse_args()


def build_env_args(ns: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        headless=bool(ns.headless),
        num_bodies=1,
        video=bool(ns.video or ns.save_snapshot),
        log=bool(ns.log),
        wandb=False,
        dock_site="dock_orion_port_a",
        dock_approach_offset=None,
    )


def parse_controllers(spec: str) -> list[str]:
    allowed = {"pd", "nominal_mpc"}
    controllers = [s.strip().lower() for s in spec.split(",") if s.strip()]
    if not controllers:
        raise ValueError("No controllers specified.")
    invalid = [c for c in controllers if c not in allowed]
    if invalid:
        raise ValueError(f"Unsupported controller(s): {invalid}")
    return controllers


def build_trial_scenarios(args: argparse.Namespace) -> list[TrialScenario]:
    random.seed(args.seed)
    np.random.seed(args.seed)

    scenarios: list[TrialScenario] = []
    failure_count = max(1, int(args.failure_count))
    thruster_pool = list(range(12))
    for trial_id in range(args.trials):
        init_pos, init_att = randomize_initial_state()
        failed_thrusters: list[int] | None = None
        if args.failure_mode != "nominal":
            if args.failure_thruster is not None:
                primary = int(args.failure_thruster)
                remaining_pool = [t for t in thruster_pool if t != primary]
                n_extra = max(0, min(failure_count - 1, len(remaining_pool)))
                sampled_extra = random.sample(remaining_pool, k=n_extra)
                failed_thrusters = [primary] + sampled_extra
            else:
                k = max(1, min(failure_count, len(thruster_pool)))
                failed_thrusters = random.sample(thruster_pool, k=k)
        scenarios.append(
            TrialScenario(
                trial_id=trial_id,
                init_pos=[float(v) for v in init_pos],
                init_att=[float(v) for v in init_att],
                failure_time=(None if args.failure_mode == "nominal" else float(args.failure_time)),
                failure_thrusters=failed_thrusters,
            )
        )
    return scenarios


def create_controller(name: str, env: AstrobeeEnv, planner: MissionPlanner):
    if name == "pd":
        env.env_cfg.control.PD.gains.Kp_x = float(env._benchmark_pd_kp_x)
        env.env_cfg.control.PD.gains.Kd_x = float(env._benchmark_pd_kd_x)
        env.env_cfg.control.PD.gains.Kp_q = float(env._benchmark_pd_kp_q)
        env.env_cfg.control.PD.gains.Kd_q = float(env._benchmark_pd_kd_q)
        return PDController(env, planner)
    if name == "nominal_mpc":
        return NominalMPCController(env, planner)
    raise ValueError(f"Unknown controller: {name}")


def _progress_fraction(planner: MissionPlanner) -> float:
    if hasattr(planner, "_intermediate_reference") and planner._intermediate_reference:
        denom = max(1, len(planner._intermediate_reference))
        return min(1.0, float(planner.idx_reference_point) / float(denom))
    denom = max(1, len(planner.waypoints))
    return min(1.0, float(planner.idx_reference_point) / float(denom))


def _inject_failure(
    *,
    env: AstrobeeEnv,
    mode: str,
    thrusters: list[int] | None,
    failure_time: float,
) -> None:
    if not thrusters:
        return
    if mode == "stuck_off":
        for thruster in thrusters:
            env.perturbations.perturbations[0].stuck_off_thruster(
                index=int(thruster),
                start_time=failure_time,
            )
    elif mode == "stuck_on":
        for thruster in thrusters:
            env.perturbations.perturbations[1].stuck_on_thruster(
                index=int(thruster),
                start_time=failure_time,
            )


def _extract_waypoint_positions(waypoint_source: Any) -> np.ndarray:
    """
    Return waypoint positions as an (N, 3) float array.
    Supports MissionPlanner Waypoint objects and plain array/list formats.
    """
    if waypoint_source is None:
        return np.zeros((0, 3), dtype=float)

    positions: list[np.ndarray] = []
    for wp in waypoint_source:
        if hasattr(wp, "position"):
            pos = np.asarray(getattr(wp, "position"), dtype=float).reshape(-1)
        else:
            pos = np.asarray(wp, dtype=float).reshape(-1)

        if pos.size >= 3:
            positions.append(pos[:3].copy())

    if not positions:
        return np.zeros((0, 3), dtype=float)
    return np.vstack(positions)


def run_single_trial(
    *,
    controller_name: str,
    scenario: TrialScenario,
    args: argparse.Namespace,
    out_dir: Path,
) -> tuple[TrialMetrics, TrialTrace]:
    env = AstrobeeEnv(args=build_env_args(args))
    env.env_cfg.sim.max_sim_time = float(args.max_sim_time)
    # Benchmark-specific PD overrides are attached on env for controller creation.
    env._benchmark_pd_kp_x = float(args.pd_kp_x)
    env._benchmark_pd_kd_x = float(args.pd_kd_x)
    env._benchmark_pd_kp_q = float(args.pd_kp_q)
    env._benchmark_pd_kd_q = float(args.pd_kd_q)

    planner = MissionPlanner(
        env,
        spacing=float(args.planner_spacing),
        clearance_dist=float(args.planner_clearance_dist),
        planner_mode="Intermediate Waypoint Tracking",
    )
    # Snapshot-only mode should not accumulate renderer frames each control step,
    # otherwise long runs can consume large amounts of memory and get OOM-killed.
    if bool(args.save_snapshot) and not bool(args.video):
        planner._visualize_renderer = lambda *args, **kwargs: None
        env.frames = []
    controller = create_controller(controller_name, env, planner)
    if controller_name == "nominal_mpc":
        # Avoid MuJoCo renderer crashes from per-step MPC prediction overlay.
        # Benchmark snapshots are generated separately via _save_multiview_snapshots.
        controller._visualize_prediction_renderer = lambda *args, **kwargs: None

    env.reset_to_state(
        pos=np.asarray(scenario.init_pos, dtype=float),
        att=np.asarray(scenario.init_att, dtype=float),
    )
    env.reset_perturbations()

    if scenario.failure_time is not None:
        _inject_failure(
            env=env,
            mode=args.failure_mode,
            thrusters=scenario.failure_thrusters,
            failure_time=scenario.failure_time,
        )

    control_dt = env.env_cfg.sim.dt * env.env_cfg.control.control_decimation
    lateral_errors: list[float] = []
    distances: list[float] = []
    times: list[float] = []
    xyz: list[np.ndarray] = []
    effort = 0.0
    snapshot_saved = False

    def _save_multiview_snapshots(trajectory_so_far: np.ndarray) -> None:
        if env.renderer is None:
            print(
                f"[inspection] snapshots skipped for {controller_name}/trial {scenario.trial_id}: "
                "renderer is unavailable"
            )
            return
        view_count = max(1, int(args.snapshot_views))
        snapshot_dir = (
            out_dir
            / "snapshots"
            / f"{controller_name}_trial{scenario.trial_id:03d}"
        )
        snapshot_dir.mkdir(parents=True, exist_ok=True)

        gateway_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gateway_full")
        if gateway_id >= 0:
            lookat = np.asarray(env.data.xpos[gateway_id], dtype=float).copy()
        else:
            lookat = np.zeros(3, dtype=float)

        # Save and restore camera state so regular rendering logic is unaffected.
        orig_azimuth = float(env.cam.azimuth)
        orig_elevation = float(env.cam.elevation)
        orig_distance = float(env.cam.distance)
        orig_lookat = np.asarray(env.cam.lookat, dtype=float).copy()

        azimuths = np.linspace(-180.0, 180.0, view_count, endpoint=False)
        elevations = np.array([-20.0, -10.0, 0.0, 10.0, 20.0], dtype=float)

        traj = np.asarray(trajectory_so_far, dtype=float).reshape(-1, 3)
        if traj.shape[0] > 200:
            idx = np.linspace(0, traj.shape[0] - 1, num=200, dtype=int)
            traj = traj[idx]

        waypoint_source = None
        if hasattr(planner, "_intermediate_reference") and planner._intermediate_reference:
            waypoint_source = planner._intermediate_reference
        elif hasattr(planner, "waypoints") and planner.waypoints:
            waypoint_source = planner.waypoints

        waypoints = _extract_waypoint_positions(waypoint_source)

        for idx, az in enumerate(azimuths):
            env.cam.azimuth = float(az)
            env.cam.elevation = float(elevations[idx % len(elevations)])
            env.cam.distance = max(16.0, orig_distance)
            env.cam.lookat[:] = lookat
            env.renderer.update_scene(env.data, env.cam)

            scene = env.renderer.scene
            max_extra = max(0, int(scene.maxgeom) - int(scene.ngeom))
            # Reserve one extra geom for the current smallsat marker.
            budget = max(0, max_extra - 1)
            n_wp = max(0, min(int(waypoints.shape[0]), budget))
            budget -= n_wp
            n_traj = max(0, min(int(traj.shape[0]), budget))

            for j in range(n_wp):
                scene.ngeom += 1
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.07, 0.0, 0.0]),
                    pos=np.asarray(waypoints[j], dtype=float),
                    mat=np.eye(3).reshape(-1),
                    rgba=np.array([0.1, 0.75, 1.0, 0.95]),
                )

            for j in range(n_traj):
                scene.ngeom += 1
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.05, 0.0, 0.0]),
                    pos=np.asarray(traj[j], dtype=float),
                    mat=np.eye(3).reshape(-1),
                    rgba=np.array([1.0, 0.85, 0.15, 0.95]),
                )

            # Highlight current smallsat pose.
            current_pos = np.asarray(env.data.qpos[:3], dtype=float)
            if int(scene.ngeom) < int(scene.maxgeom):
                scene.ngeom += 1
                mujoco.mjv_initGeom(
                    scene.geoms[scene.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=np.array([0.10, 0.0, 0.0]),
                    pos=current_pos,
                    mat=np.eye(3).reshape(-1),
                    rgba=np.array([1.0, 0.1, 0.1, 1.0]),
                )

            frame = env.renderer.render().copy()
            plt.imsave(snapshot_dir / f"view_{idx:02d}.png", frame)

        env.cam.azimuth = orig_azimuth
        env.cam.elevation = orig_elevation
        env.cam.distance = orig_distance
        env.cam.lookat[:] = orig_lookat

    while env.data.time <= env.env_cfg.sim.max_sim_time:
        ctrl = controller.get_control_input(env)
        env.step(input=ctrl)

        obs = env.get_obs()
        xyz.append(np.asarray(obs[:3], dtype=float))
        times.append(float(env.data.time))

        lateral = float(calc_lateral_tracking_error(obs=obs, planner=planner))
        lateral_errors.append(lateral)

        dist_wp = float(planner.distance_to_closest_waypoint(obs[:3]))
        distances.append(dist_wp)

        effort += float(np.sum(np.abs(env.data.ctrl))) * float(control_dt)

        if (
            bool(args.save_snapshot)
            and not snapshot_saved
            and float(env.data.time) >= float(args.snapshot_time)
        ):
            _save_multiview_snapshots(np.asarray(xyz, dtype=float))
            snapshot_saved = True

    progress = _progress_fraction(planner)
    success = int(progress >= float(args.success_progress))

    trace = TrialTrace(
        controller=controller_name,
        trial_id=scenario.trial_id,
        time=times,
        x=[float(p[0]) for p in xyz],
        y=[float(p[1]) for p in xyz],
        z=[float(p[2]) for p in xyz],
        distance_to_waypoint=distances,
    )

    metrics = TrialMetrics(
        controller=controller_name,
        trial_id=scenario.trial_id,
        success=success,
        progress_fraction=float(progress),
        mean_lateral_error=float(np.mean(lateral_errors)) if lateral_errors else float("nan"),
        rms_lateral_error=float(np.sqrt(np.mean(np.square(lateral_errors))))
        if lateral_errors
        else float("nan"),
        control_effort=float(effort),
        mean_distance_to_waypoint=float(np.mean(distances)) if distances else float("nan"),
        final_distance_to_waypoint=float(distances[-1]) if distances else float("nan"),
        sim_time=float(env.data.time),
    )

    if args.log and hasattr(env, "logger"):
        env.logger.save_log()
    env.close()

    return metrics, trace


def save_metrics_csv(path: Path, rows: list[TrialMetrics]) -> None:
    fieldnames = list(asdict(rows[0]).keys()) if rows else []
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def aggregate_metrics(rows: list[TrialMetrics]) -> dict[str, Any]:
    if not rows:
        return {}

    by_controller: dict[str, list[TrialMetrics]] = {}
    for row in rows:
        by_controller.setdefault(row.controller, []).append(row)

    summary: dict[str, Any] = {}
    for controller, values in by_controller.items():
        success = np.asarray([v.success for v in values], dtype=float)
        mean_lateral = np.asarray([v.mean_lateral_error for v in values], dtype=float)
        effort = np.asarray([v.control_effort for v in values], dtype=float)
        progress = np.asarray([v.progress_fraction for v in values], dtype=float)

        summary[controller] = {
            "n_trials": int(len(values)),
            "task_success_rate": float(np.mean(success)),
            "mean_lateral_tracking_error": float(np.mean(mean_lateral)),
            "mean_control_effort": float(np.mean(effort)),
            "mean_progress_fraction": float(np.mean(progress)),
        }

    return summary


def plot_trajectory_overlay(
    out_path: Path,
    traces: list[TrialTrace],
    scenarios: list[TrialScenario],
) -> None:
    if not traces:
        return

    plt.figure(figsize=(8, 6))
    for tr in traces:
        alpha = 0.55 if tr.trial_id == 0 else 0.25
        lw = 1.8 if tr.trial_id == 0 else 0.9
        label = f"{tr.controller} (trial 0)" if tr.trial_id == 0 else None
        plt.plot(tr.x, tr.y, alpha=alpha, lw=lw, label=label)

    for sc in scenarios:
        plt.scatter(sc.init_pos[0], sc.init_pos[1], marker="x", s=20, alpha=0.4, c="k")

    plt.xlabel("x [m]")
    plt.ylabel("y [m]")
    plt.title("Inspection Trajectory Overlay (x-y)")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def plot_distance_with_fault_annotations(
    out_path: Path,
    traces: list[TrialTrace],
    failure_mode: str,
    failure_time: float | None,
) -> None:
    if not traces:
        return

    plt.figure(figsize=(9, 5))
    for tr in traces:
        label = f"{tr.controller} (trial {tr.trial_id})" if tr.trial_id == 0 else None
        alpha = 0.6 if tr.trial_id == 0 else 0.2
        plt.plot(tr.time, tr.distance_to_waypoint, alpha=alpha, lw=1.3, label=label)

    if failure_mode != "nominal" and failure_time is not None:
        plt.axvline(failure_time, color="red", linestyle="--", lw=1.6)
        plt.text(
            failure_time,
            plt.ylim()[1] * 0.95,
            f"{failure_mode} injected",
            color="red",
            fontsize=9,
            ha="left",
            va="top",
        )

    plt.xlabel("time [s]")
    plt.ylabel("distance to closest waypoint [m]")
    plt.title("Distance-to-Target With Fault Annotation")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_path.with_suffix(".png"), dpi=180)
    plt.savefig(out_path.with_suffix(".pdf"))
    plt.close()


def main() -> None:
    args = parse_args()
    controllers = parse_controllers(args.controllers)
    scenarios = build_trial_scenarios(args)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    all_metrics: list[TrialMetrics] = []
    all_traces: list[TrialTrace] = []

    for controller in controllers:
        for scenario in scenarios:
            print(
                f"[inspection] controller={controller} trial={scenario.trial_id} "
                f"mode={args.failure_mode}"
            )
            metrics, trace = run_single_trial(
                controller_name=controller,
                scenario=scenario,
                args=args,
                out_dir=out_dir,
            )
            all_metrics.append(metrics)
            all_traces.append(trace)

    metrics_csv = out_dir / "trial_metrics.csv"
    save_metrics_csv(metrics_csv, all_metrics)

    summary = {
        "config": {
            "controllers": controllers,
            "trials": int(args.trials),
            "seed": int(args.seed),
            "max_sim_time": float(args.max_sim_time),
            "failure_mode": args.failure_mode,
            "failure_time": None if args.failure_mode == "nominal" else float(args.failure_time),
            "failure_thruster": args.failure_thruster,
            "failure_count": int(args.failure_count),
            "success_progress": float(args.success_progress),
            "save_snapshot": bool(args.save_snapshot),
            "snapshot_time": float(args.snapshot_time),
            "snapshot_views": int(args.snapshot_views),
        },
        "aggregate": aggregate_metrics(all_metrics),
        "scenarios": [asdict(s) for s in scenarios],
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    # Plot only a manageable subset to keep figures readable.
    traces_for_plot = [
        tr
        for tr in all_traces
        if tr.trial_id < min(3, args.trials)
    ]
    plot_trajectory_overlay(
        out_path=out_dir / "trajectory_overlay",
        traces=traces_for_plot,
        scenarios=scenarios,
    )
    plot_distance_with_fault_annotations(
        out_path=out_dir / "distance_to_target",
        traces=traces_for_plot,
        failure_mode=args.failure_mode,
        failure_time=(None if args.failure_mode == "nominal" else float(args.failure_time)),
    )

    print("[inspection] complete")
    print(f"[inspection] metrics: {metrics_csv}")
    print(f"[inspection] summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
