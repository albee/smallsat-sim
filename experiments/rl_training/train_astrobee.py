import os

# Set flags to improve XLA performance on GPU
os.environ["XLA_FLAGS"] = "--xla_gpu_triton_gemm_any=True "

import time

from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle_rl import OraclePlannerRL


# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnvVectorized(args=args)

# Create planner
planner = OraclePlannerRL(env, radius=0.0)

# Create runner
runner = OnPolicyRunner(env, planner)

# Pretraining
runner.pretrain()

# Learning
runner.learn()

# Adaptation module training
runner.train_adaptation_module_on_policy()

# Evaluation
runner.evaluate()

# Create simulation video if desired
env.get_sim_rendering(env.env_name)

# Save log if logging is enabled
if args.log:
    env.logger.save_log()
