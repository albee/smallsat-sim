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

            # Run ID
            rl_run_id = 0

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
                steps_per_epoch = 512
                epochs = 100
                max_ep_len = 512
                gamma = 0.99
                lam = 0.97
                actor_lr = 3e-4
                critic_lr = 3e-5
                actor_training_epochs = 3
                critic_training_epochs = 3

            class PPO:
                steps_per_epoch = 512
                epochs = 400  # Covers the nominal+curriculum phases when failures are enabled
                max_ep_len = 512  # TODO: sanity check that this duration is appropriate (and change deployment_len accordingly)
                gamma = 0.99
                lam = 0.95
                actor_lr = 3e-4  # Between 2 and 4e-4
                critic_lr = 1e-3  # >= actor_lr
                entropy_coef = 3e-5  # Make sure std/log_std steadily decreases long enough (see W&B)
                actor_training_epochs = 3
                critic_training_epochs = 3
                actor_critic_training_epochs = 3
                num_minibatches = 8
                clip_ratio = 0.2
                target_kl = 0.005
                use_value_clip = True
                value_clip_coef = 0.2
                debug_prints = False

            # Sequential failure curriculum knobs
            curriculum_nominal_epochs = 100
            curriculum_phase_epochs = 50
            curriculum_failure_fraction = 0.4  # 60% nominal, 40% failures
            curriculum_disturbance_fraction = 0.1
            curriculum_critic_warmup_epochs = 15
            curriculum_critic_warmup_scale = 1.0 / 3.0
            curriculum_eval_interval = 5

            # Mission tolerances (tuned to current training scenario - do not change)
            sigma_pos = 3.0
            sigma_vel = 3.0  # sigma_pos / (N_settle * dt)
            sigma_att = 6.0  # radians
            sigma_angvel = 6.0  # sigma_att / (N_settle * dt)

            # Reward weights (can play with overall magnitude s.t. critic loss is well-behaved)
            w_pos, w_vel, w_att, w_angvel = 5e1, 0.0, 1e1, 0.0  # Disregard vel and angvel terms for now

            # Penalty weights
            lam_fuel = 1e-4
            lam_speed_terminal = 1e-1
            lam_ang_speed_terminal = 1e-3
            lam_fuel_terminal = 5e-4
            lam_wrench_residual = 0.0  # Disregard wrench residual for now, minimizing it is implicitly encoded in the reward
            wrench_residual_tolerance = 1e-1
            wrench_residual_clip = 0.5

            # Context window length for the adaptation module
            context_window_len = 50

            # Hyperparams for the adaptation module training (NOTE: have not been tuned yet + CNN/transformer may require different sets)
            am_epochs = 100
            am_lr = 3e-4
            am_weight_decay = 0.05
            am_grad_clip_norm = 1.0
            am_kl_weight = 0.01

            # Hyperparams for the evaluation loop
            episode_len = 512
            n_evals = 10

            # Hyperparams for the controller
            deployment_len = 512  # Set to None to disable

            # Regression testing of functional rollout vs legacy rollout
            verify_functional_rollout = False
            verify_functional_rollout_atol = 1e-4
            verify_functional_rollout_rtol = 1e-3

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
