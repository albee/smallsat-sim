from smallsat_sim.planners.base_planner import BasePlanner

import numpy as np


class OraclePlanner(BasePlanner):
    def __init__(self, radius=20, spacing=0.25) -> None:
        super().__init__()
        self.radius = radius
        self.spacing = spacing

        # generate a circular reference trajectory around the gateway
        self._generate_reference()

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Dedicated method which is called externally
        """
        return self.reference_points[0]

    def _generate_reference(self):
        """
        Generates a circle in the yz plane around the gateway
        """
        self.reference_points = []
        for i in range(int((2 * np.pi * self.radius) // self.spacing)):
            x = 0  # x-coordinate remains constant as the circle is in the yz-plane
            y = self.radius * np.sin(
                i * 2 * np.pi * self.radius / self.spacing
            )  # y-coordinate
            z = self.radius * np.cos(
                i * 2 * np.pi * self.radius / self.spacing
            )  # z-coordinate

            self.reference_points.append(np.array([x, y, z]))
