import time
import os

from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL
from smallsat_sim.planners.mission.mission import MissionPlanner


# Set flags to improve XLA performance on GPU
os.environ['XLA_FLAGS'] = (
    '--xla_gpu_enable_triton_softmax_fusion=true '
    '--xla_gpu_triton_gemm_any=True '
    # '--xla_gpu_enable_async_collectives=true '
    # '--xla_gpu_enable_latency_hiding_scheduler=true '
    # '--xla_gpu_enable_highest_priority_async_stream=true '
)

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnvVectorized(args=args)

# Create planner
planner = OraclePlannerRL(env)

# Create runner
runner = OnPolicyRunner(env, planner)

# Pretraining
runner.pretrain()

# Learning
runner.learn()

# Evaluation
runner.evaluate()

# Create simulation video if desired
env.get_sim_rendering(env.env_name)

# Save log if logging is enabled
if args.log:
    env.logger.save_log()
