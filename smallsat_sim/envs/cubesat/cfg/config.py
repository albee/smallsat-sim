from smallsat_sim.envs.base_env_config import BaseEnvConfig

import numpy as np


class Body:
    def __init__(self, name, pos=[0, 0, 0], euler=[0, 0, 0]) -> None:
        self.name = name
        self.pos = pos
        self.euler = euler


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
            bodies_list.append(Body(name=f"body{i}", pos=[i, 0, 10]))

    # Holds all information for the controller in use
    class control:

        # PD controller parameters
        class PD:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 10

            class gains:
                Kp_x = 1.0
                Kd_x = 1.0
                Kp_q = 0.05
                Kd_q = 0.1

        # LQR controller params
        class LQR:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 25

            class cost:
                # Intermediate quadratic cost on state
                Q = np.zeros((13, 13))
                Q[10, 10] = 1e-2
                Q[11, 11] = 1e-2
                Q[12, 12] = 1e-2

                # Intermediate cost on input
                R = 1e-3 * np.eye(12)

                # Terminal quadratic cost on position
                Q_e = np.eye(3)

        # Nominal MPC controller parameters
        class NominalMPC:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 25
            Ts = BaseEnvConfig.sim.dt * control_decimation

            # Define MPC's horizon
            N = 80

            class cost:
                # Intermediate quadratic cost on state
                Q = np.zeros((13, 13))
                Q[10, 10] = 1e-2
                Q[11, 11] = 1e-2
                Q[12, 12] = 1e-2

                # Intermediate cost on input
                R = 1e-3 * np.eye(12)

                # Terminal quadratic cost on position
                Q_e = np.eye(3)

        # RL controller params
        class RL:
            control_decimation = 25

            num_envs = 40

            # TODO: add RL params here
