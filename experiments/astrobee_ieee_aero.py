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

import faulthandler  # to catch mujoco viewer issues


# Get arguments for script execution
args = get_args()
faulthandler.enable()

class Mode(Enum):
  LINGER = 1
  FLYBY = 2
  FAILURE_MANY = 3
  FAILURE_SINGLE = 4

# Change this to one of the 3 sim cases we are testing
# Note that all plotting is in smallsat-sim/misc/ieee_aero.ipynb
mode = Mode.LINGER

# Create environment
env = AstrobeeEnv(args=args)
env.env_cfg.sim.max_sim_time = 1000.0

# Linger mode
# NB: has a segfault issue!
if (mode == Mode.LINGER):
  planner = MissionPlanner(env, planner_mode="Waypoint Tracking")
  ctrl = NominalMPCController(env, planner)
# Flyby mode
elif (mode == Mode.FLYBY):
  planner = MissionPlanner(env, planner_mode="Intermediate Waypoint Tracking")
  # hack for accurate waypoint counter
  # @alex would be interesting to think of other "waypoint reached" verifications for MPCC,
  # perhaps breaking through a plane normal to the trajectory?
  # very liklely we want to keep track of "progress" along the path, perhaps could
  # just take the quantity that's already computed in the MPCC cost function
  planner.clearance_dist = 0.6
  ctrl = NominalMPCCController(env, planner)
# Flyby mode, failure
elif (mode == Mode.FAILURE_MANY):
  env.perturbations.perturbations[0].stuck_off_thruster(0, 0.0)
  env.perturbations.perturbations[0].stuck_off_thruster(2, 0.0)
  env.perturbations.perturbations[0].stuck_off_thruster(7, 0.0)
  env.perturbations.perturbations[0].stuck_off_thruster(9, 0.0)
  planner = MissionPlanner(env, planner_mode="Intermediate Waypoint Tracking")
  ctrl = NominalMPCCController(env, planner)
elif (mode == Mode.FAILURE_SINGLE):
  env.perturbations.perturbations[0].stuck_off_thruster(0, 0.0)
  planner = MissionPlanner(env, planner_mode="Intermediate Waypoint Tracking")
  planner.clearance_dist = 0.8
  ctrl = NominalMPCCController(env, planner)

# Define start time
start_time = time.time()

# Simulation loop
failed = False
last_ref_point = planner.idx_reference_point-1
while (env.data.time <= env.env_cfg.sim.max_sim_time and
       planner.idx_reference_point < 21):
    real_time = time.time() - start_time

    sim_time = env.data.time

    # manually stop
    if (mode == Mode.FAILURE_MANY and
        sim_time > 30.0):
      break

    # Printout of progress along inspection points
    # print(planner.idx_reference_point)
    if last_ref_point is not planner.idx_reference_point:
      last_ref_point = planner.idx_reference_point
      print("Reference point: ", planner.idx_reference_point, " sim_time: ", sim_time, " [s]")

    if True:
        # Calculate control action (open-loop)
        ctrl_input = ctrl.get_control_input(env)

        # Advance simulation
        env.step(input=ctrl_input)

print("Simulation complete!")

# Create simulation video if desired
env.get_sim_rendering(env.env_name)

# Save log if logging is enabled
if args.log:
    env.logger.save_log()

# Close the environment to avoid viewer/renderer issues
env.close()

print("Simulation complete.")
