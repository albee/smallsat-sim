import numpy as np
import random

from smallsat_sim.envs.base_env_config import BaseEnvConfig


class Body:
    def __init__(self, name, pos=[0, 0, 0], euler=[0, 0, 0]) -> None:
        self.name = name
        self.pos = pos
        self.euler = euler


class EnvConfig(BaseEnvConfig):
    """Environment configuration class. Contains dynamic vehicle information."""

    model = "astrobee"
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

            # NOTE: the following four flags can be overwritten when initializing AstrobeeEnvVectorized

            # Enable random failures during training by default
            train_with_failures = True

            # Pretrain the actor and critic?
            use_pretrained = False

            # Use adaptive approach? If False, the state is not being augmented
            use_adaptive_approach = True

            # Adaptation module architecture ("cnn" or "transformer")
            am_architecture = "transformer"

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
                epochs = 80
                max_ep_len = 1024
                gamma = 0.99
                lam = 0.95
                actor_lr = 5e-4
                critic_lr = 5e-3
                entropy_coef = 1e-3

            # Mission tolerances
            sigma_pos = 0.2
            sigma_vel = 1.0  # 5 * sigma_pos / (N_settle * dt)
            sigma_att = 0.2  # radians
            sigma_angvel = 1.0  # 5 * sigma_att / (N_settle * dt)

            # Reward weights
            w_pos, w_vel, w_att, w_angvel = 1.5, 0.5, 1.0, 0.3

            # Penalty weights
            lam_fuel = 0.1
            lam_speed_terminal = 0.1
            lam_ang_speed_terminal = 0.1
            lam_fuel_terminal = 0.05
            lam_wrench_residual = 0.05
            wrench_residual_tolerance = 0.05
            wrench_residual_clip = 2.0

            # Context window length for the adaptation module
            context_window_len = 50

            # Hyperparams for the adaptation module training
            am_lr = 1e-3
            am_weight_decay = 0.0
            am_grad_clip_norm = 1.0
            am_kl_weight = 0.0

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
