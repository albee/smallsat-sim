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
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.envs.cubesat.env import CubesatEnv
from smallsat_sim.planners.mission.docking import DockingPlanner
from smallsat_sim.utils.helpers import calc_attitude_error


def _normalized(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("Vector norm too small for normalization.")
    return v / n


@dataclass
class DockingTrialMetrics:
    trial_id: int
    rendezvous_success: int
    docking_success: int
    precontact_reached: int
    precontact_time: float
    contact_detected: int
    first_contact_time: float
    settled_contact: int
    bounded_contact_forces: int
    max_contact_force: float
    final_position_error: float
    final_attitude_error: float
    control_effort: float
    sim_time: float


@dataclass
class TrialSnapshotMeta:
    trial_id: int
    contact_time: float | None


@dataclass
class TrialTrace:
    trial_id: int
    time: list[float]
    position_error: list[float]
    attitude_error: list[float]
    contact_force: list[float]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Docking surrogate experiment with randomized initial relative pose and "
            "soft-contact stability metrics."
        )
    )
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--seed", type=int, default=11)
    parser.add_argument(
        "--vehicle",
        type=str,
        choices=("astrobee", "cubesat"),
        default="astrobee",
        help="Vehicle/environment to evaluate in the same docking surrogate setup.",
    )
    parser.add_argument("--dock-site", type=str, default="dock_orion_interface_a")
    parser.add_argument("--approach-offset", type=float, default=2.0)
    parser.add_argument(
        "--dock-surface-offset",
        type=float,
        default=0.0,
        help=(
            "Shift final dock setpoint along -approach_axis [m]. "
            "Positive values move target inward toward station surface."
        ),
    )
    parser.add_argument("--initial-radius-min", type=float, default=1.0)
    parser.add_argument("--initial-radius-max", type=float, default=3.0)
    parser.add_argument("--initial-z-span", type=float, default=1.0)
    parser.add_argument("--max-initial-att-deg", type=float, default=25.0)
    parser.add_argument("--switch-distance", type=float, default=0.35)
    parser.add_argument("--dock-distance", type=float, default=0.15)
    parser.add_argument("--precontact-pos-tol", type=float, default=0.30)
    parser.add_argument("--precontact-att-tol", type=float, default=0.35)
    parser.add_argument("--precontact-deadline", type=float, default=80.0)
    parser.add_argument("--settled-pos-tol", type=float, default=0.25)
    parser.add_argument("--settled-speed-tol", type=float, default=0.10)
    parser.add_argument("--contact-force-bound", type=float, default=8.0)
    parser.add_argument("--max-sim-time", type=float, default=140.0)
    parser.add_argument(
        "--stop-on-contact",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Terminate a trial immediately once first contact is detected.",
    )
    parser.add_argument(
        "--post-contact-timeout",
        type=float,
        default=12.0,
        help=(
            "If > 0, terminate this many sim-seconds after first contact. "
            "Ignored when --stop-on-contact is enabled."
        ),
    )
    parser.add_argument(
        "--headless",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-snapshots",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--log",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default="experiments/results/docking_surrogate_contact",
    )
    return parser.parse_args()


def build_env_args(args: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        headless=bool(args.headless),
        num_bodies=1,
        video=bool(args.save_snapshots),
        log=bool(args.log),
        wandb=False,
        dock_site=str(args.dock_site),
        dock_approach_offset=float(args.approach_offset),
    )


def dock_site_map(env_cfg) -> dict[str, dict]:
    dock_sites = getattr(getattr(env_cfg, "gateway", None), "dock_sites", [])
    return {site["name"]: site for site in dock_sites}


def site_id_or_raise(env, site_name: str) -> int:
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id >= 0:
        return site_id

    available = []
    for idx in range(env.model.nsite):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, idx)
        if name is not None:
            available.append(name)
    raise ValueError(f"Dock site '{site_name}' not found. Available: {available}")


def random_initial_pose(
    *,
    rng: random.Random,
    pre_dock_pos: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    theta = rng.uniform(0.0, 2.0 * np.pi)
    r = rng.uniform(float(args.initial_radius_min), float(args.initial_radius_max))
    z = rng.uniform(-float(args.initial_z_span), float(args.initial_z_span))

    pos = pre_dock_pos + np.array([r * np.cos(theta), r * np.sin(theta), z], dtype=float)

    max_att_rad = np.deg2rad(float(args.max_initial_att_deg))
    att = np.array(
        [
            rng.uniform(-max_att_rad, max_att_rad),
            rng.uniform(-max_att_rad, max_att_rad),
            rng.uniform(-max_att_rad, max_att_rad),
        ],
        dtype=float,
    )
    return pos, att


def _capture_contact_viewpoints(
    env: AstrobeeEnv,
    *,
    lookat: np.ndarray,
    n_views: int = 10,
) -> list[np.ndarray]:
    if env.renderer is None:
        return []

    original_azimuth = float(env.cam.azimuth)
    original_elevation = float(env.cam.elevation)
    original_distance = float(env.cam.distance)
    original_lookat = np.asarray(env.cam.lookat, dtype=float).copy()

    # Fixed ring of camera poses around the docking contact point.
    azimuths = np.linspace(-180.0, 180.0, num=n_views, endpoint=False)
    elevations = np.array([-10.0, 5.0, 15.0, 25.0, 10.0, 0.0, -5.0, 20.0, 30.0, 12.0], dtype=float)
    distances = np.array([4.6, 4.8, 5.0, 5.2, 4.4, 4.9, 5.3, 4.7, 5.1, 4.5], dtype=float)

    if elevations.shape[0] < n_views:
        elevations = np.resize(elevations, n_views)
    if distances.shape[0] < n_views:
        distances = np.resize(distances, n_views)

    frames: list[np.ndarray] = []
    try:
        for i in range(n_views):
            env.cam.lookat[:] = np.asarray(lookat, dtype=float)
            env.cam.azimuth = float(azimuths[i])
            env.cam.elevation = float(elevations[i])
            env.cam.distance = float(distances[i])
            env.renderer.update_scene(env.data, env.cam)
            frames.append(env.renderer.render().copy())
    finally:
        env.cam.azimuth = original_azimuth
        env.cam.elevation = original_elevation
        env.cam.distance = original_distance
        env.cam.lookat[:] = original_lookat

    return frames


def _contact_force_between_bodies(
    env,
    body_a: int,
    body_b: int,
) -> tuple[bool, float]:
    any_contact = False
    max_force = 0.0

    for ci in range(int(env.data.ncon)):
        contact = env.data.contact[ci]
        g1 = int(contact.geom1)
        g2 = int(contact.geom2)
        b1 = int(env.model.geom_bodyid[g1])
        b2 = int(env.model.geom_bodyid[g2])

        if {b1, b2} != {body_a, body_b}:
            continue

        force6 = np.zeros(6, dtype=float)
        mujoco.mj_contactForce(env.model, env.data, ci, force6)
        force_mag = float(np.linalg.norm(force6[:3]))
        max_force = max(max_force, force_mag)
        any_contact = True

    return any_contact, max_force


def run_trial(
    *,
    env,
    controller: NominalMPCController,
    planner: DockingPlanner,
    dock_position: np.ndarray,
    dock_attitude: np.ndarray,
    chaser_body_id: int,
    gateway_body_id: int,
    trial_id: int,
    rng: random.Random,
    args: argparse.Namespace,
) -> tuple[DockingTrialMetrics, TrialTrace, TrialSnapshotMeta, dict[str, Any]]:
    planner.stage = 0

    init_pos, init_att = random_initial_pose(rng=rng, pre_dock_pos=planner.pre_dock_position, args=args)
    env.reset_to_state(pos=init_pos, att=init_att)

    precontact_reached = False
    precontact_time = np.nan
    contact_detected = False
    first_contact_time = np.nan
    max_contact_force = 0.0

    snapshot_meta = TrialSnapshotMeta(
        trial_id=trial_id,
        contact_time=None,
    )
    contact_view_frames: list[np.ndarray] = []

    control_dt = env.env_cfg.sim.dt * env.env_cfg.control.control_decimation
    control_effort = 0.0

    times: list[float] = []
    position_errors: list[float] = []
    attitude_errors: list[float] = []
    contact_forces: list[float] = []

    while env.data.time <= float(args.max_sim_time):
        ctrl = controller.get_control_input(env)
        env.step(input=ctrl)
        control_effort += float(np.sum(np.abs(env.data.ctrl))) * float(control_dt)

        obs = env.get_obs()
        pos_error = float(np.linalg.norm(obs[:3] - dock_position))
        att_error = float(calc_attitude_error(dock_attitude, obs[3:7]))
        speed = float(np.linalg.norm(obs[7:10]))

        has_contact, contact_force = _contact_force_between_bodies(
            env,
            body_a=chaser_body_id,
            body_b=gateway_body_id,
        )

        if has_contact:
            max_contact_force = max(max_contact_force, contact_force)
            if not contact_detected:
                contact_detected = True
                first_contact_time = float(env.data.time)
                if args.save_snapshots and not contact_view_frames:
                    contact_view_frames = _capture_contact_viewpoints(
                        env,
                        lookat=dock_position,
                        n_views=10,
                    )
                    snapshot_meta.contact_time = float(env.data.time)

        if (
            not precontact_reached
            and float(env.data.time) <= float(args.precontact_deadline)
            and pos_error <= float(args.precontact_pos_tol)
            and att_error <= float(args.precontact_att_tol)
        ):
            precontact_reached = True
            precontact_time = float(env.data.time)

        times.append(float(env.data.time))
        position_errors.append(pos_error)
        attitude_errors.append(att_error)
        contact_forces.append(contact_force if has_contact else 0.0)

        # Termination logic after first contact.
        if contact_detected:
            if bool(args.stop_on_contact):
                break
            post_contact_timeout = float(args.post_contact_timeout)
            if post_contact_timeout > 0.0 and float(env.data.time) >= (
                first_contact_time + post_contact_timeout
            ):
                break

    final_obs = env.get_obs()
    final_pos_error = float(np.linalg.norm(final_obs[:3] - dock_position))
    final_att_error = float(calc_attitude_error(dock_attitude, final_obs[3:7]))
    final_speed = float(np.linalg.norm(final_obs[7:10]))

    settled_contact = int(
        contact_detected
        and final_pos_error <= float(args.settled_pos_tol)
        and final_speed <= float(args.settled_speed_tol)
    )

    rendezvous_success = int(precontact_reached)
    bounded_forces = int(max_contact_force <= float(args.contact_force_bound))
    docking_success = int(
        precontact_reached and contact_detected and settled_contact and bounded_forces
    )

    metrics = DockingTrialMetrics(
        trial_id=trial_id,
        rendezvous_success=rendezvous_success,
        docking_success=docking_success,
        precontact_reached=int(precontact_reached),
        precontact_time=float(precontact_time) if precontact_reached else float("nan"),
        contact_detected=int(contact_detected),
        first_contact_time=float(first_contact_time) if contact_detected else float("nan"),
        settled_contact=settled_contact,
        bounded_contact_forces=bounded_forces,
        max_contact_force=float(max_contact_force),
        final_position_error=float(final_pos_error),
        final_attitude_error=float(final_att_error),
        control_effort=float(control_effort),
        sim_time=float(env.data.time),
    )

    trace = TrialTrace(
        trial_id=trial_id,
        time=times,
        position_error=position_errors,
        attitude_error=attitude_errors,
        contact_force=contact_forces,
    )

    return metrics, trace, snapshot_meta, {"contact_views": contact_view_frames}


def save_metrics_csv(path: Path, rows: list[DockingTrialMetrics]) -> None:
    if not rows:
        return
    fieldnames = list(asdict(rows[0]).keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


def aggregate(rows: list[DockingTrialMetrics]) -> dict[str, float | int]:
    if not rows:
        return {}
    arr = lambda key: np.asarray([getattr(r, key) for r in rows], dtype=float)
    return {
        "n_trials": int(len(rows)),
        "rendezvous_success_rate": float(np.mean(arr("rendezvous_success"))),
        "docking_success_rate": float(np.mean(arr("docking_success"))),
        "mean_final_position_error": float(np.mean(arr("final_position_error"))),
        "mean_final_attitude_error": float(np.mean(arr("final_attitude_error"))),
        "mean_control_effort": float(np.mean(arr("control_effort"))),
        "max_contact_force_observed": float(np.max(arr("max_contact_force"))),
    }


def plot_timeseries(out_dir: Path, traces: list[TrialTrace]) -> None:
    if not traces:
        return

    plt.figure(figsize=(9, 5))
    for tr in traces:
        alpha = 0.6 if tr.trial_id == 0 else 0.2
        label = f"trial {tr.trial_id}" if tr.trial_id == 0 else None
        plt.plot(tr.time, tr.position_error, alpha=alpha, label=label)
    plt.xlabel("time [s]")
    plt.ylabel("position error [m]")
    plt.title("Docking Position Error")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "position_error.png", dpi=180)
    plt.savefig(out_dir / "position_error.pdf")
    plt.close()

    plt.figure(figsize=(9, 5))
    for tr in traces:
        alpha = 0.6 if tr.trial_id == 0 else 0.2
        label = f"trial {tr.trial_id}" if tr.trial_id == 0 else None
        plt.plot(tr.time, tr.contact_force, alpha=alpha, label=label)
    plt.xlabel("time [s]")
    plt.ylabel("contact force magnitude [N]")
    plt.title("Contact Force Sequence")
    plt.grid(True, alpha=0.2)
    plt.legend(loc="best")
    plt.tight_layout()
    plt.savefig(out_dir / "contact_force.png", dpi=180)
    plt.savefig(out_dir / "contact_force.pdf")
    plt.close()

    # Additional "pretty" variants for figures (no title, no trial labels).
    pretty_label_fontsize = 24
    pretty_tick_fontsize = 20

    plt.figure(figsize=(9, 5))
    for tr in traces:
        alpha = 1.0
        plt.plot(tr.time, tr.position_error, color="#343aeb", alpha=alpha, lw=1.8)
    plt.xlabel("time [s]", fontsize=pretty_label_fontsize)
    plt.ylabel("dock position error [m]", fontsize=pretty_label_fontsize)
    plt.grid(True, alpha=0.18)
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=pretty_tick_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "position_error_pretty.png", dpi=220, bbox_inches="tight")
    plt.savefig(out_dir / "position_error_pretty.pdf", bbox_inches="tight")
    plt.close()

    plt.figure(figsize=(9, 5))
    for tr in traces:
        alpha = 1.0
        plt.plot(tr.time, tr.contact_force, color="#343aeb", alpha=alpha, lw=1.8)
    plt.xlabel("time [s]", fontsize=pretty_label_fontsize)
    plt.ylabel("contact force [N]", fontsize=pretty_label_fontsize)
    plt.grid(True, alpha=0.18)
    ax = plt.gca()
    ax.tick_params(axis="both", which="major", labelsize=pretty_tick_fontsize)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    plt.tight_layout()
    plt.savefig(out_dir / "contact_force_pretty.png", dpi=220, bbox_inches="tight")
    plt.savefig(out_dir / "contact_force_pretty.pdf", bbox_inches="tight")
    plt.close()


def save_contact_view_snapshots(
    out_dir: Path,
    trial_id: int,
    frames: list[np.ndarray],
    contact_time: float | None,
) -> None:
    if not frames:
        return
    trial_dir = out_dir / "contact_snapshots" / f"trial_{trial_id:03d}"
    trial_dir.mkdir(parents=True, exist_ok=True)
    for idx, frame in enumerate(frames):
        plt.imsave(trial_dir / f"contact_view_{idx:02d}.png", frame)
    if contact_time is not None:
        with (trial_dir / "meta.json").open("w", encoding="utf-8") as f:
            json.dump({"trial_id": int(trial_id), "contact_time": float(contact_time)}, f, indent=2)


def main() -> None:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    rng = random.Random(args.seed)

    out_dir = Path(args.out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.vehicle == "astrobee":
        env = AstrobeeEnv(args=build_env_args(args))
    else:
        env = CubesatEnv(args=build_env_args(args))
    env.env_cfg.sim.max_sim_time = float(args.max_sim_time)

    site_cfg_map = dock_site_map(env.env_cfg)
    if args.dock_site not in site_cfg_map:
        raise ValueError(
            f"Unknown dock site '{args.dock_site}'. Available: {list(site_cfg_map.keys())}"
        )

    site_cfg = site_cfg_map[args.dock_site]
    site_id = site_id_or_raise(env, args.dock_site)
    dock_site_position = np.asarray(env.data.site_xpos[site_id], dtype=float).copy()
    dock_attitude = np.asarray(site_cfg.get("quat", [1.0, 0.0, 0.0, 0.0]), dtype=float)

    approach_axis = _normalized(np.asarray(site_cfg.get("approach_axis", [1.0, 0.0, 0.0]), dtype=float))
    # Positive surface offset shifts terminal target inward along -approach_axis.
    dock_position = dock_site_position - approach_axis * float(args.dock_surface_offset)
    approach_offset = float(args.approach_offset)
    pre_dock_position = dock_position + approach_axis * approach_offset

    print(
        "[docking] setup "
        f"site={args.dock_site} "
        f"dock_site={dock_site_position.tolist()} "
        f"dock_target={dock_position.tolist()} "
        f"pre_dock={pre_dock_position.tolist()} "
        f"approach_axis={approach_axis.tolist()} "
        f"dock_surface_offset={float(args.dock_surface_offset):.3f}m"
    )

    planner = DockingPlanner(
        env=env,
        pre_dock_position=pre_dock_position,
        dock_position=dock_position,
        dock_attitude=dock_attitude,
        switch_distance=float(args.switch_distance),
        dock_distance=float(args.dock_distance),
    )
    controller = NominalMPCController(env, planner)
    if bool(args.save_snapshots):
        # Snapshot mode captures explicit frames only; disable per-step renderer
        # frame accumulation from planner/controller visualization hooks to avoid OOM.
        planner._visualize_renderer = lambda *args, **kwargs: None
        controller._visualize_prediction_renderer = lambda *args, **kwargs: None
        env.frames = []

    chaser_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "body0")
    gateway_body_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_BODY, "gateway_full")
    if chaser_body_id < 0 or gateway_body_id < 0:
        raise RuntimeError("Could not resolve body ids for contact analysis.")

    trial_metrics: list[DockingTrialMetrics] = []
    traces: list[TrialTrace] = []
    for trial_id in range(args.trials):
        print(f"[docking] trial={trial_id}/{args.trials - 1}")
        metrics, trace, snap_meta, snapshots = run_trial(
            env=env,
            controller=controller,
            planner=planner,
            dock_position=dock_position,
            dock_attitude=dock_attitude,
            chaser_body_id=chaser_body_id,
            gateway_body_id=gateway_body_id,
            trial_id=trial_id,
            rng=rng,
            args=args,
        )
        trial_metrics.append(metrics)
        traces.append(trace)
        if args.save_snapshots:
            save_contact_view_snapshots(
                out_dir=out_dir,
                trial_id=trial_id,
                frames=list(snapshots.get("contact_views", [])),
                contact_time=snap_meta.contact_time,
            )

    save_metrics_csv(out_dir / "trial_metrics.csv", trial_metrics)
    plot_timeseries(out_dir, traces)

    summary = {
        "config": {
            "trials": int(args.trials),
            "seed": int(args.seed),
            "vehicle": str(args.vehicle),
            "dock_site": str(args.dock_site),
            "approach_offset": float(args.approach_offset),
            "dock_surface_offset": float(args.dock_surface_offset),
            "max_sim_time": float(args.max_sim_time),
            "precontact_pos_tol": float(args.precontact_pos_tol),
            "precontact_att_tol": float(args.precontact_att_tol),
            "precontact_deadline": float(args.precontact_deadline),
            "contact_force_bound": float(args.contact_force_bound),
            "stop_on_contact": bool(args.stop_on_contact),
            "post_contact_timeout": float(args.post_contact_timeout),
            "settled_pos_tol": float(args.settled_pos_tol),
            "settled_speed_tol": float(args.settled_speed_tol),
        },
        "aggregate": aggregate(trial_metrics),
        "trial_metrics": [asdict(m) for m in trial_metrics],
    }

    with (out_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    if args.log and hasattr(env, "logger"):
        env.logger.save_log()

    env.close()
    print("[docking] complete")
    print(f"[docking] summary: {out_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
