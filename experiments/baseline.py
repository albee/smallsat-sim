from smallsat_sim.controllers.open_loop.open_loop import OpenLoop
from smallsat_sim.envs.baseline.baseline_env import BaselineEnv
from smallsat_sim.utils.helpers import get_args

import time

# Get arguments for script execution
args = get_args()

# Create environment
env = BaselineEnv()

# Create controller
ctrl = OpenLoop()

# Define start time
start_time = time.time()

# Simulation loop
while True:
    
    real_time = time.time() - start_time

    sim_time = env.data.time

    if sim_time < real_time:
    # Take step in environment
    # Controller callback is executed internally
        env.step()
        