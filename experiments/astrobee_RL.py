import time

from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized


# from smallsat_sim.planners.oracle.oracle import OraclePlanner

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnvVectorized(args=args)

# Create planner
# planner = OraclePlanner(env)

# Create runner
runner = OnPolicyRunner(env)

# Simulation loop
runner.control()

# Create simulation video if desired
env.get_sim_rendering(env.env_name)