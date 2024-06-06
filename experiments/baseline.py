from smallsat_sim.controllers.open_loop.controller import OpenLoopController
from smallsat_sim.envs.baseline.env import BaselineEnv
from smallsat_sim.utils.helpers import get_args

import time

# Get arguments for script execution
args = get_args()

# Create environment
env = BaselineEnv(args=args)

# Create controller
ctrl = OpenLoopController()

# Define start time
start_time = time.time()

# Simulation loop
while env.data.time <= env.env_cfg.sim.sim_time:
    
    real_time = time.time() - start_time

    sim_time = env.data.time

    if sim_time < real_time:
        # Calculate control action (open-loop)
        ctrl_input = ctrl.get_control_input(env)
        
        # Advance simulation
        env.step(input=ctrl_input)