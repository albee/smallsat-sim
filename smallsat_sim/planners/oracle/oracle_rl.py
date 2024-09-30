from smallsat_sim.planners.base_planner import BasePlanner

import numpy as np


class OraclePlannerRL(BasePlanner):
    def __init__(self, env, radius=5.0, spacing=0.5, clearance_dist=0.2) -> None:
        super().__init__(env)
        self.radius = radius
        self.spacing = spacing
        self.clearance_dist = clearance_dist

        # Initialize current reference point
        self.idx_reference_point = 0

        # generate a circular reference trajectory around the gateway
        self._generate_reference()

        # visualize the circle
        self.visualize(self.reference_points)

    def get_reference(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Dedicated method which is called externally
        """
        # Check if current state is close enough
        dist = np.linalg.norm(
            obs[0:3]
            - self.reference_points[
                self.idx_reference_point % len(self.reference_points)
            ]
        )

        # If smallsat is closer than the clearance distance, the next reference point is queried
        if dist < self.clearance_dist:
            self.idx_reference_point += 1

        # Visualize the waypoints in the saved video
        self._visualize_renderer(self.reference_points)

        return (
            self.reference_points[
                self.idx_reference_point % len(self.reference_points)
            ].reshape(3, 1),
            np.array([1, 0, 0, 0]).reshape(4, 1),
        )

    def closest_point_on_trajectory(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Finds the closest point on the trajectory to the given point.
        Returns the position and corresponding arc length.
        """
        raise NotImplementedError

    def _generate_reference(self):
        """
        Generates a circle in the yz plane around the gateway
        """
        self.reference_points = []
        for i in range(int((2 * np.pi * self.radius) // self.spacing)):
            x = self.radius * np.cos(i * self.spacing / self.radius)
            y = self.radius * np.sin(i * self.spacing / self.radius)
            z = 10.17  # z-coordinate remains constant as the circle is in the xy-plane

            self.reference_points.append(np.array([x, y, z]))
