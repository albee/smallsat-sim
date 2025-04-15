import time
import os
import jax.numpy as jnp

from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.rl.controller import RLController
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.planners.mission.mission import MissionPlanner


# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = (
    "--xla_gpu_enable_triton_softmax_fusion=true " "--xla_gpu_triton_gemm_any=True "
)

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnvVectorized(
    args=args, init_pos=jnp.array([5.0, 0.0, 10.17]), max_start_offset=0.5
)

# Create planner
planner = OraclePlannerRL(env)

# Create controller
ctrl = RLController(env, planner)

# Simulation loop
ctrl.control()

# Create simulation video if desired
env.get_sim_rendering(env.env_name)

# Save log if logging is enabled
if args.log:
    env.logger.save_log()
