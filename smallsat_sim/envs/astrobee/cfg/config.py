from smallsat_sim.envs.base_env_config import BaseEnvConfig
from smallsat_sim.planners.mission.mission import MissionPlanner
import numpy as np
import random


class Body:
    def __init__(self, name, pos=[0, 0, 0], euler=[0, 0, 90]) -> None:
        self.name = name
        self.pos = pos
        self.euler = euler


def randomize_initial_state() -> tuple[np.ndarray, np.ndarray]:
    """
    Hardcoded for MissionPlanner at the moment.
    
    """
    # Define positional references
    positions = [
        [-3.3, -9, 0],         # Point 1
        [-3.3, -18, 0],        # Point 2
        [-1, -20, 5],          # Point 3
        [16, 0, 23],           # Point 4
        [16, 0, 0],            # Point 5
        [16, 0, -23],          # Point 6
        [16, 0, -26],          # Point 7
        [7, 0, -26],           # Point 8
        [7, 0, -4.5],          # Point 9
        [3, 0, -4.5],          # Point 10
        [3, 0, -8],            # Point 11
        [3.6, 16, -8],         # Point 12
        [3.6, 16, 0],          # Point 13
        [-2.5, 16, 0],         # Point 14
        [-1.5, 12, -2],        # Point 15
        [-2.0, 8, 0], 
        [-2.0, 6.0, 0],         # Point 16 
        [-4.0, 5.5, 0],        # Point 17
        [-10, 5, 0],          # Point 18
        [-18, 10, 0],          # Point 19
        [-18, 0, 0],           # Point 20
        [-18, 0, -5],          # Point 21
        [-8, 0, -5],           # Point 22
        [-3.3, 0, -3.5],       # Point 23
        [-3.3, -9, -3.5],      # Point 24
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
    #idx = 12

    pos = positions[idx]
    att = attitudes[idx]

    test = np.random.uniform(-0.5, 0.5, 3)
    test = np.random.randint(-45, 46, 3)

    pos += np.random.uniform(-0.5, 0.5, 3)
    att += np.random.randint(-45, 46, 3)

    return pos, att

    pos += np.random.uniform(-0.7, 0.7, 3)
    att += np.random.randint(-60, 60, 3)


    return pos.tolist(), att.tolist()


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
                R = 1e-2 * np.eye(12)

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
                R = 1e-2 * np.eye(12)

                # Cost on lag error
                q_l = 5e-2

                # Contouring cost
                Q_c = 9e-2 * np.eye(3)

                # Cost on angular velocity
                Q_omega = 1e-2 * np.eye(3)

                # Reward for progress
                q_theta = 5e-2

                # Penalty on attitude error
                Q_q = 8e-3 * np.eye(4)

        # GPMPC controller parameters
        class GPMPC:
            # Decimate the controller frequency such that it doesn't
            # run equally fast to the simulation discretization
            control_decimation = 50
            Ts = BaseEnvConfig.sim.dt * control_decimation

            # Define MPC's horizon
            N = 80

            class cost:
                # Intermediate cost on input
                R = 1e-2 * np.eye(12)

                # Cost on lag error
                q_l = 5e-2

                # Contouring cost
                Q_c = 9e-2 * np.eye(3)

                # Cost on angular velocity
                Q_omega = 1e-2 * np.eye(3)

                # Reward for progress
                q_theta = 5e-2

                # Penalty on attitude error
                Q_q = 8e-3 * np.eye(4)

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
