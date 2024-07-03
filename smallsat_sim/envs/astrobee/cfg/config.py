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
            bodies_list.append(Body(name=f"body{i}", pos=[i, 0.0, 10.5]))

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

        # LQR controller params
        class LQR:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 25

            class cost:
                # Intermediate quadratic cost on state
                Q = np.zeros((13,13))
                Q[10,10] = 1e-2
                Q[11,11] = 1e-2
                Q[12,12] = 1e-2

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
                # Intermediate quadratic cost on state wrt. artificial reference
                Q = np.zeros((13, 13))

                # Penalize position
                Q[0, 0] = 1e-1
                Q[1, 1] = 1e-1
                Q[2, 2] = 1e-1

                # Penalize attitude
                Q[3, 3] = 1e-1
                Q[4, 4] = 1e-1
                Q[5, 5] = 1e-1
                Q[6, 6] = 1e-1

                # Penalize linear velocity
                Q[7, 7] = 1e-1
                Q[8, 8] = 1e-1
                Q[9, 9] = 1e-1

                # Penalize angular velocity
                Q[10, 10] = 1e-1
                Q[11, 11] = 1e-1
                Q[12, 12] = 1e-1

                # Intermediate cost on input
                R = 1e-4 * np.eye(12)

                # Quadratic cost on artificial reference wrt. reference point
                T = 2 * Q

                # Terminal quadratic cost on position
                Q_e = np.eye(3)

        # GPMPC controller parameters
        class GPMPC:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 50
            Ts = BaseEnvConfig.sim.dt * control_decimation

            # Define MPC's horizon
            N = 80

            class cost:
                # Intermediate quadratic cost on state wrt. artificial reference
                Q = np.zeros((13, 13))

                # Penalize position
                Q[0, 0] = 1e-1
                Q[1, 1] = 1e-1
                Q[2, 2] = 1e-1

                # Penalize attitude
                Q[3, 3] = 1e-1
                Q[4, 4] = 1e-1
                Q[5, 5] = 1e-1
                Q[6, 6] = 1e-1

                # Penalize linear velocity
                Q[7, 7] = 1e-1
                Q[8, 8] = 1e-1
                Q[9, 9] = 1e-1

                # Penalize angular velocity
                Q[10, 10] = 1e-1
                Q[11, 11] = 1e-1
                Q[12, 12] = 1e-1

                # Intermediate cost on input
                R = 1e-4 * np.eye(12)

                # Quadratic cost on artificial reference wrt. reference point
                T = 2 * Q


    class planner:
        resolution = 1  # Resolution of the grid
        epsilon = 2.5  # Initial heuristic inflation factor
        epsilon_increment = 0.2 # Increment of the heuristic inflation factor
        epsilon_decrement = 0.2 # Decrement of the heuristic inflation factor
        bounds = np.array([[-10,-10,-10], [10, 10, 10]])  # Bounds for the planner
        # Define the possible directions and their costs
        unit_directions = {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1, \
                           (-1, 0, 0): 1, (0, -1, 0): 1, (0, 0, -1): 1, \
                           (1, 1, 0): np.sqrt(2), (1, 0, 1): np.sqrt(2), (0, 1, 1): np.sqrt(2), \
                           (-1, -1, 0): np.sqrt(2), (-1, 0, -1): np.sqrt(2), (0, -1, -1): np.sqrt(2), \
                           (1, -1, 0): np.sqrt(2), (-1, 1, 0): np.sqrt(2), (1, 0, -1): np.sqrt(2), \
                           (-1, 0, 1): np.sqrt(2), (0, 1, -1): np.sqrt(2), (0, -1, 1): np.sqrt(2), \
                           (1, 1, 1): np.sqrt(3), (-1, -1, -1): np.sqrt(3), \
                           (1, -1, -1): np.sqrt(3), (-1, 1, -1): np.sqrt(3), (-1, -1, 1): np.sqrt(3), \
                           (1, 1, -1): np.sqrt(3), (1, -1, 1): np.sqrt(3), (-1, 1, 1): np.sqrt(3)}
        
        start_pos = (0, 0, 10)
        goal_pos = (0, 3, -10)

