import time
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.controllers.nominal_mpc.controller import NominalMPCController
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.planners.oracle.oracle import OraclePlanner
import glfw

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnv(args=args)

# Create planner
planner = OraclePlanner(env)

# Create controller
ctrl = PDController(env, planner)

window = glfw.create_window(1920, 1080, "MJX Simulation", None, None)
if not window:
    glfw.terminate()
    raise Exception("GLFW window could not be created!")
        
glfw.make_context_current(window)

# Define start time
start_time = time.time()

# Simulation loop
# while True:
while not glfw.window_should_close(window):

    real_time = time.time() - start_time

    sim_time = env.data.time

    if True:
        # Calculate control action (open-loop)
        ctrl_input = ctrl.get_control_input(env)

        # Advance simulation
        env.step(args=args, input=ctrl_input, window=window)

glfw.terminate()