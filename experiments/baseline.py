from smallsat_sim.controllers.open_loop.open_loop import OpenLoop
from smallsat_sim.envs.baseline.baseline_env import BaselineEnv
from smallsat_sim.utils.helpers import get_args

# Get arguments for script execution
args = get_args()

# Create environment
env = BaselineEnv()

# Create controller
ctrl = OpenLoop()

