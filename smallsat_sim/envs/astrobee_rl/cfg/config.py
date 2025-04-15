import numpy as np
import random

from smallsat_sim.envs.base_env_config import BaseEnvConfig


class Body:
    def __init__(self, name, pos=[0, 0, 0], euler=[0, 0, 0]) -> None:
        self.name = name
        self.pos = pos
        self.euler = euler


def randomize_initial_state() -> tuple[np.ndarray, np.ndarray]:
    """
    Hardcoded for MissionPlanner at the moment.

    """
    # Define positional references
    positions = [
        [-3.3, -9, 0],  # Point 1
        [-3.3, -18, 0],  # Point 2
        [-1, -20, 5],  # Point 3
        [16, 0, 23],  # Point 4
        [16, 0, 0],  # Point 5
        [16, 0, -23],  # Point 6
        [16, 0, -26],  # Point 7
        [7, 0, -26],  # Point 8
        [7, 0, -4.5],  # Point 9
        [3, 0, -4.5],  # Point 10
        [3, 0, -8],  # Point 11
        [3.6, 16, -8],  # Point 12
        [3.6, 16, 0],  # Point 13
        [-2.5, 16, 0],  # Point 14
        [-1.5, 12, -2],  # Point 15
        [-2.0, 8, 0],
        [-2.0, 6.0, 0],  # Point 16
        [-4.0, 5.5, 0],  # Point 17
        [-10, 5, 0],  # Point 18
        [-18, 10, 0],  # Point 19
        [-18, 0, 0],  # Point 20
        [-18, 0, -5],  # Point 21
        [-8, 0, -5],  # Point 22
        [-3.3, 0, -3.5],  # Point 23
        [-3.3, -9, -3.5],  # Point 24
    ]

    # Define attitude references (Euler angles)
    attitudes = [
        [0, 0, 90],  # Point 1
        [0, 0, 90],  # Point 2
        [0, 0, 90],  # Point 3
        [0, 0, 180],  # Point 4
        [0, 0, 180],  # Point 5
        [0, 0, 180],  # Point 6
        [0, 0, 180],  # Point 7
        [0, -90, 0],  # Point 8
        [0, 0, 0],  # Point 9
        [0, -90, 0],  # Point 10
        [0, -180, 0],  # Point 11
        [0, -90, 0],  # Point 12
        [90, 0, -90],  # Point 13
        [0, 0, -90],  # Point 14
        [0, 0, -90],  # Point 15
        [0, 0, -90],
        [0, 0, -90],  # Point 16
        [0, 0, -90],  # Point 17
        [0, 0, -90],  # Point 18
        [0, 0, -45],  # Point 19
        [0, 0, -45],  # Point 20
        [0, -45, 0],  # Point 21
        [0, -90, 0],  # Point 22
        [0, -90, 0],  # Point 23
        [0, -45, 90],  # Point 24
    ]

    # Define random integer
    idx = random.randint(0, 24)
    idx = random.randint(11, 17)
    idx = 17

    pos = positions[idx]
    att = attitudes[idx]

    # test = np.random.uniform(-0.5, 0.5, 3)
    # test = np.random.randint(-45, 46, 3)

    # pos += np.random.uniform(-0.5, 0.5, 3)
    # att += np.random.randint(-45, 46, 3)

    # return pos, att

    pos += np.random.uniform(-0.7, 0.7, 3)
    att += np.random.randint(-60, 60, 3)

    return pos.tolist(), att.tolist()


class EnvConfig(BaseEnvConfig):
    """Environment configuration class. Contains dynamic vehicle information."""

    model = "cubesat"
    compiler = {"convexhull": "true"}
    visual = {
        "headlight": {
            "ambient": "0.5 0.5 0.5",
            "specular": "0.5 0.5 0.5",
            "diffuse": "0.5 0.5 0.5",
        }
    }

    class Bodies:
        num_bodies = 1
        bodies_list = []
        for i in range(num_bodies):
            bodies_list.append(Body(name=f"body{i}", pos=[i, 0.0, 10.17]))

        max_start_offset = 3.0  # Maximum offset from the initial position at each reset

    # Holds all information for the controller in use
    class control:

        # PD controller parameters
        class PD:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 10

            class gains:
                Kp_x = 0.2
                Kd_x = 1.0
                Kp_q = 3.0
                Kd_q = 5.0

        # RL controller params
        class RL:
            control_decimation = 25

            # Number of environments
            num_envs = 2048

            # NOTE: the following two flags can be overwritten when initializing AstrobeeEnvVectorized

            # Pretrain the actor and critic?
            use_pretrained = True

            # Use adaptation module?
            use_adaptive_approach = True

            # Hyperparams for the learning loop
            class VPG:
                steps_per_epoch = 4096
                epochs = 50
                max_ep_len = 1024
                gamma = 0.99
                lam = 0.97
                actor_lr = 1e-4
                critic_lr = 1e-3

            class PPO:
                steps_per_epoch = 4096
                epochs = 50
                max_ep_len = 1024
                gamma = 0.99
                lam = 0.95
                actor_lr = 5e-4
                critic_lr = 5e-3

            # Hyperparams for the adaptation module
            am_lr = 1e-3

            # Hyperparams for the evaluation loop
            episode_len = 700
            n_evals = 10

            # Hyperparams for the controller
            deployment_len = 700  # Set to None to disable

    class planner:
        resolution = 1  # Resolution of the grid
        epsilon = 2.5  # Initial heuristic inflation factor
        epsilon_increment = 0.2  # Increment of the heuristic inflation factor
        epsilon_decrement = 0.2  # Decrement of the heuristic inflation factor
        bounds = np.array([[-10, -10, -10], [10, 10, 10]])  # Bounds for the planner
        # Define the possible directions and their costs
        unit_directions = {
            (1, 0, 0): 1,
            (0, 1, 0): 1,
            (0, 0, 1): 1,
            (-1, 0, 0): 1,
            (0, -1, 0): 1,
            (0, 0, -1): 1,
            (1, 1, 0): np.sqrt(2),
            (1, 0, 1): np.sqrt(2),
            (0, 1, 1): np.sqrt(2),
            (-1, -1, 0): np.sqrt(2),
            (-1, 0, -1): np.sqrt(2),
            (0, -1, -1): np.sqrt(2),
            (1, -1, 0): np.sqrt(2),
            (-1, 1, 0): np.sqrt(2),
            (1, 0, -1): np.sqrt(2),
            (-1, 0, 1): np.sqrt(2),
            (0, 1, -1): np.sqrt(2),
            (0, -1, 1): np.sqrt(2),
            (1, 1, 1): np.sqrt(3),
            (-1, -1, -1): np.sqrt(3),
            (1, -1, -1): np.sqrt(3),
            (-1, 1, -1): np.sqrt(3),
            (-1, -1, 1): np.sqrt(3),
            (1, 1, -1): np.sqrt(3),
            (1, -1, 1): np.sqrt(3),
            (-1, 1, 1): np.sqrt(3),
        }

        start_pos = (0, 0, 10)
        goal_pos = (0, 3, -10)
