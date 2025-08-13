import time
import numpy as np
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.mpcc.controller_2d import NominalMPCCController2D
from smallsat_sim.envs.astrobee_2d.env import Astrobee2DEnv
from smallsat_sim.planners.mission.mission_2d import MissionPlanner2D

from smallsat_sim.communication.udp import create_communication_channels
import time

# Get arguments for script execution
args = get_args()

# Create environment
env = Astrobee2DEnv(args=args)

# Create planner
planner = MissionPlanner2D(env)

# Create controller
ctrl = NominalMPCCController2D(env, planner)
ctrl_input = ctrl.get_control_input(env)

# Define start time
start_time = time.time()

receiver, publisher = create_communication_channels()

# Simulation loop
while env.data.time <= env.env_cfg.sim.max_sim_time:

    real_time = time.time() - start_time

    sim_time = env.data.time

    if True:
        # Calculate control action (open-loop)
        observations = env.obs
        # publish observation
        publisher.publish_state(observations[0:3], observations[3:7], observations[7:10], observations[10:13])

        # get command
        ctrl_input = receiver.get_control_input()
        env.step(input=ctrl_input)

# Create simulation video if desired
env.get_sim_rendering(env.env_name)

# Save log if logging is enabled
if args.log:
    env.logger.save_log()
