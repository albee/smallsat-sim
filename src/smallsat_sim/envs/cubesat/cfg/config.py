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

    class gateway:
        # Keep dock-site schema aligned with Astrobee env so docking scripts can be
        # reused across vehicles without special cases.
        dock_sites = [
            {
                "name": "dock_orion_port_a",
                "pos": [-13.8, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "approach_axis": [-1.0, 0.0, 0.0],
                "approach_offset": 2.0,
                "size": [0.20],
                "rgba": [1.0, 0.2, 0.2, 0.8],
            },
            {
                # Orion-to-Gateway docking interface (NDS-like region), centered closer
                # to the adapter tunnel than the engine-bell side.
                "name": "dock_orion_interface_a",
                "pos": [-5.95, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "approach_axis": [-1.0, 0.0, 0.0],
                "approach_offset": 2.0,
                "size": [0.20],
                "rgba": [0.2, 1.0, 0.35, 0.85],
            },
            {
                # I-HAB <-> HALO axial interface region.
                "name": "dock_halo_ihab_interface_a",
                "pos": [0.89, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "approach_axis": [1.0, 0.0, 0.0],
                "approach_offset": 2.0,
                "size": [0.20],
                "rgba": [1.0, 0.75, 0.2, 0.85],
            },
            {
                # HALO <-> PPE axial interface region.
                "name": "dock_halo_ppe_interface_a",
                "pos": [7.34, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "approach_axis": [1.0, 0.0, 0.0],
                "approach_offset": 2.0,
                "size": [0.20],
                "rgba": [0.95, 0.55, 0.15, 0.85],
            },
            {
                # Crew airlock side interface candidate.
                "name": "dock_crew_airlock_a",
                "pos": [-3.11, -6.97, -0.02],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "approach_axis": [0.0, -1.0, 0.0],
                "approach_offset": 2.0,
                "size": [0.20],
                "rgba": [0.4, 0.9, 1.0, 0.85],
            },
            {
                # HLS side interface candidate.
                "name": "dock_hls_side_a",
                "pos": [3.62, 7.82, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "approach_axis": [0.0, 1.0, 0.0],
                "approach_offset": 2.0,
                "size": [0.20],
                "rgba": [0.65, 1.0, 0.65, 0.85],
            },
            {
                # Logistics side interface candidate.
                "name": "dock_logistics_side_a",
                "pos": [3.69, -10.03, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "approach_axis": [0.0, -1.0, 0.0],
                "approach_offset": 2.0,
                "size": [0.20],
                "rgba": [0.75, 0.85, 1.0, 0.85],
            },
            {
                "name": "dock_ppe_port_a",
                "pos": [13.2, 0.0, 0.0],
                "quat": [1.0, 0.0, 0.0, 0.0],
                "approach_axis": [1.0, 0.0, 0.0],
                "approach_offset": 2.0,
                "size": [0.20],
                "rgba": [0.2, 0.7, 1.0, 0.8],
            },
        ]

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

                # Quadratic cost on artificial reference wrt. reference point
                # (required by NominalMPCController).
                T = 1.5 * Q

                # Terminal quadratic cost on position
                Q_e = np.eye(3)
