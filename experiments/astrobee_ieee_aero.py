import time
from enum import Enum
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.controllers.nominal_mpc.controller import NominalMPCController
from smallsat_sim.controllers.mpcc.controller import NominalMPCCController
from smallsat_sim.controllers.lqr.controller import LQRController
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.planners.mission.mission import MissionPlanner

# Get arguments for script execution
args = get_args()

class Mode(Enum):
  LINGER = 1
  FLYBY = 2
  FAILURE = 3

mode = Mode.FLYBY

# Create environment
env = AstrobeeEnv(args=args)
env.env_cfg.sim.max_sim_time = 200.0

# Linger mode
# NB: has a segfault issue!
if (mode == Mode.LINGER):
  planner = MissionPlanner(env, planner_mode="Waypoint Tracking")
  ctrl = NominalMPCController(env, planner)
# Flyby mode
elif (mode == Mode.FLYBY):
  planner = MissionPlanner(env, planner_mode="Intermediate Waypoint Tracking")
  ctrl = NominalMPCCController(env, planner)
# Flyby mode, failure
elif (mode == Mode.FAILURE):
  planner = MissionPlanner(env, planner_mode="Intermediate Waypoint Tracking")
  ctrl = NominalMPCCController(env, planner)

# Define start time
start_time = time.time()

# Simulation loop
failed = False
while env.data.time <= env.env_cfg.sim.max_sim_time:
    real_time = time.time() - start_time

    sim_time = env.data.time

    if (mode == Mode.FAILURE and sim_time > 20.0):
      failed = True

    if True:
        # Calculate control action (open-loop)
        ctrl_input = ctrl.get_control_input(env)

        # Advance simulation
        env.step(input=ctrl_input)

# Create simulation video if desired
env.get_sim_rendering(env.env_name)

# Save log if logging is enabled
if args.log:
    env.logger.save_log()