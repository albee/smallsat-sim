import time
import numpy as np
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.mpcc.controller import NominalMPCCController
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.planners.mission.mission import MissionPlanner

# from smallsat_sim.communication.udp_receiver import UDPReceiver
from smallsat_sim.communication.udp import create_communication_channels

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnv(args=args)

# Create planner
planner = MissionPlanner(env)

# Create controller
ctrl = NominalMPCCController(env, planner)
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
