import time
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.envs.astrobee.env import AstrobeeEnv

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnv(args=args)

# Create controller
ctrl = PDController()

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
        
