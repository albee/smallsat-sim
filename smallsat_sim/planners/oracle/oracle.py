from smallsat_sim.planners.base_planner import BasePlanner

import numpy as np


class OraclePlanner(BasePlanner):
    def __init__(self, env, radius=9.5, spacing=1, clearance_dist=0.2) -> None:
        super().__init__(env)
        self.radius = radius
        self.spacing = spacing
        self.clearance_dist = clearance_dist

        # Initialize current reference point
        self.idx_reference_point = 0

        # generate a circular reference trajectory around the gateway
        self._generate_reference()

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Dedicated method which is called externally
        """
        # Check if current state is close enough
        dist = np.linalg.norm(
            obs[0:3] - self.reference_points[self.idx_reference_point]
        )

        # If smallsat is closer than the clearance distance, the next reference point is queried
        if dist < self.clearance_dist:
            self.idx_reference_point += 1

        # Visualizes the next 3 points
        self.visualize(
            [
                self.reference_points[
                    self.idx_reference_point % len(self.reference_points)
                ],
                self.reference_points[
                    (self.idx_reference_point + 1) % len(self.reference_points)
                ],
                self.reference_points[
                    (self.idx_reference_point + 2) % len(self.reference_points)
                ],
            ]
        )

        return self.reference_points[self.idx_reference_point]

    def _generate_reference(self):
        """
        Generates a circle in the yz plane around the gateway
        """
        self.reference_points = []
        for i in range(int((2 * np.pi * self.radius) // self.spacing)):
            x = 0  # x-coordinate remains constant as the circle is in the yz-plane
            y = self.radius * np.sin(i * self.spacing / self.radius)  # y-coordinate
            z = self.radius * np.cos(i * self.spacing / self.radius)  # z-coordinate

            self.reference_points.append(np.array([x, y, z]))
