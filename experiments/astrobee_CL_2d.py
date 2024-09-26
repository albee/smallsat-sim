import time
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.mpcc.controller_2d import NominalMPCCController2D
from smallsat_sim.envs.astrobee_2d.env import Astrobee2DEnv
from smallsat_sim.planners.mission.mission_2d import MissionPlanner2D

# Get arguments for script execution
args = get_args()

# Create environment
env = Astrobee2DEnv(args=args)

# Create planner
planner = MissionPlanner2D(env)

# Create controller
ctrl = NominalMPCCController2D(env, planner)

# Define start time
start_time = time.time()

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