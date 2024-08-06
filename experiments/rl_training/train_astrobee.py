import time
import os

from smallsat_sim.utils.helpers import get_args
from smallsat_sim.controllers.rl.runners.on_policy_runner import OnPolicyRunner
from smallsat_sim.envs.astrobee_rl.env import AstrobeeEnvVectorized
from smallsat_sim.planners.oracle.oracle import OraclePlanner
from smallsat_sim.planners.mission.mission import MissionPlanner


# os.environ['XLA_PYTHON_CLIENT_MEM_FRACTION'] = '.50'
# os.environ['XLA_PYTHON_CLIENT_PREALLOCATE'] = 'false'

os.environ["MUJOCO_GL"] = "egl"
xla_flags = os.environ.get("XLA_FLAGS", "")
xla_flags += " --xla_gpu_triton_gemm_any=True"
os.environ["XLA_FLAGS"] = xla_flags

# Get arguments for script execution
args = get_args()

# Create environment
env = AstrobeeEnvVectorized(args=args)

# Create planner
planner = OraclePlanner(env)

# Create runner
runner = OnPolicyRunner(env, planner)

# Learning
runner.learn()

# Evaluation
runner.evaluate()

# Create simulation video if desired
env.get_sim_rendering(env.env_name)
