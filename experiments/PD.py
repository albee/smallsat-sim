import time
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.planners.oracle.oracle import OraclePlanner

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnv(args=args)

# Create planner
planner = OraclePlanner()

# Create controller
ctrl = PDController(env, planner)

# Set a reference (Tentative code for demonstration purposes)
ctrl.x_ref[0] = 1

# Define start time
start_time = time.time()

# Simulation loop
while True:
    
    real_time = time.time() - start_time

    sim_time = env.data.time

    if sim_time < real_time:
        # Calculate control action (open-loop)
        ctrl_input = ctrl.get_control_input(env)
        
        # Advance simulation
        env.step(input=ctrl_input)