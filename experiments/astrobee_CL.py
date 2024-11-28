import time
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.controllers.nominal_mpc.controller import NominalMPCController
from smallsat_sim.controllers.mpcc.controller import NominalMPCCController
from smallsat_sim.controllers.lqr.controller import LQRController
from smallsat_sim.controllers.gp_mpc.controller import GPMPC
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.planners.mission.mission import MissionPlanner

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnv(args=args)

# Create planner
planner = MissionPlanner(env, planner_mode="Waypoint Tracking")

# Create controller
ctrl = NominalMPCCController(env, planner)

# Define start time
start_time = time.time()

env.env_cfg.sim.sim_time = 100

# Simulation loop
while env.data.time <= env.env_cfg.sim.max_sim_time:

    real_time = time.time() - start_time

    sim_time = env.data.time

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

print("Simulation complete.")