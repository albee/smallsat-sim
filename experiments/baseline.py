from controllers.open_loop.open_loop import OpenLoop
from envs.baseline.baseline_env import BaselineEnv
from utils.helpers import get_args

# Get arguments for script execution
args = get_args()

# Create environment
env = BaselineEnv()

# Create controller
ctrl = OpenLoop()

