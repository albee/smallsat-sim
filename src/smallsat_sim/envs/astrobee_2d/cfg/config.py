from smallsat_sim.envs.base_env_config import BaseEnvConfig
import numpy as np
import random


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
    dist = 0.61
    astrobee_height = 0.16
    positions = np.array([
        [0, -dist, astrobee_height],
        [dist/2, -dist, astrobee_height],
        [dist, -dist, astrobee_height],
        [dist, -dist/2, astrobee_height],
        [dist, 0, astrobee_height],
        [dist, dist/2, astrobee_height],
        [dist, dist, astrobee_height],
        [dist/2, dist, astrobee_height],
        [0, dist, astrobee_height],
        [-dist/2, dist, astrobee_height],
        [-dist, dist, astrobee_height],
        [-dist, dist/2, astrobee_height],
        [-dist, 0, astrobee_height],
        [-dist, -dist/2, astrobee_height],
        [-dist, -dist, astrobee_height],
        [-dist/2, -dist, astrobee_height],
    ])

    # Define attitude references (Euler angles)
    attitudes = np.array([
        [0, 0, 0],
        [0, 0, 0],
        [0, 0, 45],
        [0, 0, 90],
        [0, 0, 90],
        [0, 0, 90],
        [0, 0, 135],
        [0, 0, 180],
        [0, 0, 180],
        [0, 0, 180],
        [0, 0, -135],
        [0, 0, -90],
        [0, 0, -90],
        [0, 0, -90],
        [0, 0, -45],
        [0, 0, 0],
    ])

    # Define random integer
    idx = random.randint(0, 15)
    idx = 12

    pos = positions[idx]
    att = attitudes[idx]

    pos = positions[0]
    att = attitudes[0]

    pos += np.random.uniform(-0.05, 0.05, 3)
    pos[-1] = 0

    return pos.tolist(), att.tolist()


class EnvConfig(BaseEnvConfig):
    """Environment configuration class. Contains dynamic vehicle information."""

    model = "astrobee_2d"
    compiler = {"convexhull": "true"}
    visual = {
        "headlight": {
            "ambient": "0.5 0.5 0.5",
            "specular": "0.5 0.5 0.5",
            "diffuse": "0.5 0.5 0.5",
        }
    }
    dof = "3d"
    class Bodies:
        num_bodies = 1
        bodies_list = []
        for i in range(num_bodies):
            pos, att = randomize_initial_state()
            bodies_list.append(
                Body(name=f"body{i}", pos=pos, euler=att)
            )

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
                Q = np.zeros((13, 13))
                Q[10, 10] = 1e-2
                Q[11, 11] = 1e-2
                Q[12, 12] = 1e-2

                # Intermediate cost on input
                R = 1e-3 * np.eye(8)

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
                R = 1e-2 * np.eye(8)

                # Quadratic cost on artificial reference wrt. reference point
                T = 1.5 * Q

                # Terminal quadratic cost on position
                Q_e = np.eye(3)

        # Nominal MPC controller parameters
        class NominalMPCC:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 50
            Ts = BaseEnvConfig.sim.dt * control_decimation

            # Define MPC's horizon
            N = 80

            class cost:
                # Intermediate cost on input
                R = 5e-2 * np.eye(8)

                # Cost on lag error
                q_l = 5e-2

                # Contouring cost
                Q_c = 9e-2 * np.eye(3)

                # Cost on angular velocity
                Q_omega = 1e-2 * np.eye(3)

                # Reward for progress
                q_theta = 5e-2

                # Penalty on attitude error
                Q_q = 1e-3 * np.eye(4)

        # GPMPC controller parameters
        class GPMPC:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 50
            Ts = BaseEnvConfig.sim.dt * control_decimation

            # Define MPC's horizon
            N = 75

            class cost:
                # Intermediate cost on input
                R = 5e-2 * np.eye(8)

                # Cost on lag error
                q_l = 5e-2

                # Contouring cost
                Q_c = 9e-2 * np.eye(3)

                # Cost on angular velocity
                Q_omega = 1e-2 * np.eye(3)

                # Reward for progress
                q_theta = 5e-2

                # Penalty on attitude error
                Q_q = 6e-3 * np.eye(4)

    class planner:
        resolution = 1  # Resolution of the grid
        epsilon = 2.5  # Initial heuristic inflation factor
        epsilon_increment = 0.2  # Increment of the heuristic inflation factor
        epsilon_decrement = 0.2  # Decrement of the heuristic inflation factor
        bounds = np.array([[-10, -10, 0], [10, 10, 0]])  # Bounds for the planner
        # Define the possible directions and their costs
        unit_directions = {
            (1, 0, 0): 1,
            (0, 1, 0): 1,
            (-1, 0, 0): 1,
            (0, -1, 0): 1,
            (1, 1, 0): np.sqrt(2),
            (-1, -1, 0): np.sqrt(2),
            # (0, -1, -1): np.sqrt(2),
            (1, -1, 0): np.sqrt(2),
            (-1, 1, 0): np.sqrt(2),
            
        }

        start_pos = (0, -0.6, 0.16)
        goal_pos = (-0.6, -0.6, 0.16)
