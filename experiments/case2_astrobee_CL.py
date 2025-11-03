import time
from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.pd.controller import PDController
from smallsat_sim.controllers.nominal_mpc.controller import NominalMPCController
from smallsat_sim.controllers.mpcc.controller import NominalMPCCController
from smallsat_sim.controllers.lqr.controller import LQRController
from smallsat_sim.controllers.gp_mpc.controller import GPMPC
from smallsat_sim.envs.astrobee.env import AstrobeeEnv
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.planners.mission.mission import MissionPlanner, MissionPlannerCubicSpline

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnv(args=args)

# Define perturbations
env.perturbations.perturbations[0].stuck_off_thruster(0, 55.0)
env.perturbations.perturbations[0].stuck_off_thruster(2, 55.0)
env.perturbations.perturbations[2].register_perturbation(1, 80, 0.0)

# Create planner
planner = MissionPlanner(env)

# Create controller
ctrl = GPMPC(env, planner)

# Define start time
start_time = time.time()

env.env_cfg.sim.sim_time = 120

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

# Close the environment to avoid viewer/renderer issues
env.close()

print("Simulation complete.")
