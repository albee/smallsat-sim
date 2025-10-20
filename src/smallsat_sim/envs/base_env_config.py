import numpy as np
import random

try:
    import torch
except ImportError:  # Torch is optional for RL-only setups
    torch = None

# Set the seed for Python's random module
random.seed(41)

# Set the seed for NumPy
np.random.seed(42)

if torch is not None:
    torch.manual_seed(42)


class BaseEnvConfig:
    """
    Base class of all environment configurations
    Holds general parameters applicable for all envs
    """

    dof = "6d"

    # Viewer options
    class viewer:
        # Standard viewer rendering @50Hz
        viewer_decimation = 10

    class sim:
        dt = 1 / 500  # simulation runs @500Hz
        max_sim_time = 200  # max. simulation time in seconds
        seed = 42  # Base seed used for JAX PRNGs unless overridden

        class noise:
            add_obs_noise = False  # add noise to ob of env
            sigma_r = 1e-2
            sigma_q = 1e-3
            sigma_v = 1e-3
            sigma_w = 1e-3

        class obs:
            v_frame = "body"  # can alternatively be switched to "inertial"

    class renderer:
        start_recording = 0.0  # Start time of the recorded window
        end_recording = 120  # Finish time of the recorded window
        width = 1920  # Width resolution
        height = 1080  # Height resolution
        fps = 30  # Frames per second of video
