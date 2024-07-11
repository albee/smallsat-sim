from smallsat_sim.planners.base_planner import BasePlanner

import numpy as np


class MissionPlanner(BasePlanner):
    def __init__(self, env) -> None:
        super().__init__(env)

        # Load the waypoints
        self._load_waypoints()

    def _load_waypoints(self) -> None:
        """
        Loads the sparse waypoints that shall be reached
        """
        # Sets list of points
        self.waypoints = [
            [0, 0, 10],
            [0, 10, 0],
            [10, 0, 0],
            [-3.4, -9, 0], # Point 1
            [-3.4, -25, 0], # Point 1
        ]

        # Visualize wapoints for DEBUG
        if True:
            self.visualize(self.waypoints, size=[0.2, 0, 0])

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns a reference point based on current observations
        """
        return np.array([0, 0, 10])
