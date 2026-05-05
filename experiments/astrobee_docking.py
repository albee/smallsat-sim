import time
import numpy as np
import mujoco

from smallsat_sim.utils.helpers import get_args
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.controllers.nominal_mpc.controller import NominalMPCController
from smallsat_sim.planners.mission.docking import DockingPlanner


def _dock_site_map(env_cfg) -> dict[str, dict]:
    dock_sites = getattr(getattr(env_cfg, "gateway", None), "dock_sites", [])
    return {site["name"]: site for site in dock_sites}


def _site_id_or_raise(env: AstrobeeEnv, site_name: str) -> int:
    site_id = mujoco.mj_name2id(env.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if site_id >= 0:
        return site_id

    available = []
    for idx in range(env.model.nsite):
        name = mujoco.mj_id2name(env.model, mujoco.mjtObj.mjOBJ_SITE, idx)
        if name is not None:
            available.append(name)
    raise ValueError(
        f"Dock site '{site_name}' not found. Available sites: {available}"
    )


def _normalized(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    if n < 1e-12:
        raise ValueError("approach_axis must be non-zero")
    return v / n


args = get_args()
env = AstrobeeEnv(args=args)

dock_site_cfg_map = _dock_site_map(env.env_cfg)
if args.dock_site not in dock_site_cfg_map:
    raise ValueError(
        f"Unknown dock site '{args.dock_site}'. "
        f"Configured sites: {list(dock_site_cfg_map.keys())}"
    )

dock_site_cfg = dock_site_cfg_map[args.dock_site]
site_id = _site_id_or_raise(env, args.dock_site)
dock_position_site = env.data.site_xpos[site_id].copy()
dock_attitude = np.asarray(
    dock_site_cfg.get("quat", [1.0, 0.0, 0.0, 0.0]),
    dtype=float,
)

approach_axis = _normalized(
    np.asarray(dock_site_cfg.get("approach_axis", [1.0, 0.0, 0.0]), dtype=float)
)
approach_offset = (
    float(args.dock_approach_offset)
    if args.dock_approach_offset is not None
    else float(dock_site_cfg.get("approach_offset", 2.0))
)
surface_offset = float(getattr(args, "dock_surface_offset", 0.0))
# Positive surface_offset moves the terminal docking target toward the station
# surface along -approach_axis (same axis, opposite direction as approach).
dock_position = dock_position_site - approach_axis * surface_offset
pre_dock_position = dock_position + approach_axis * approach_offset

planner = DockingPlanner(
    env=env,
    pre_dock_position=pre_dock_position,
    dock_position=dock_position,
    dock_attitude=dock_attitude,
    switch_distance=0.35,
    dock_distance=0.15,
)
ctrl = NominalMPCController(env, planner)

print(
    f"Docking run started. site={args.dock_site}, "
    f"dock_site={dock_position_site.tolist()}, "
    f"dock_target={dock_position.tolist()}, pre_dock={pre_dock_position.tolist()}, "
    f"surface_offset={surface_offset:.3f}m"
)

start_time = time.time()
last_print_time = -1.0
docked = False
while env.data.time <= env.env_cfg.sim.max_sim_time:
    ctrl_input = ctrl.get_control_input(env)
    env.step(input=ctrl_input)

    obs = env.get_obs()
    docked = planner.is_docked(obs)
    if docked:
        break

    # Keep console output readable in long runs.
    if env.data.time - last_print_time >= 1.0:
        last_print_time = env.data.time
        dist = np.linalg.norm(obs[:3] - dock_position)
        print(
            f"sim_time={env.data.time:.2f}s stage={planner.stage} "
            f"dist_to_dock={dist:.3f}m"
        )

wall_time = time.time() - start_time
print(
    f"Docking {'succeeded' if docked else 'did not complete'} "
    f"in sim_time={env.data.time:.2f}s wall_time={wall_time:.2f}s"
)

env.get_sim_rendering(env.env_name)
if args.log:
    env.logger.save_log()
env.close()
