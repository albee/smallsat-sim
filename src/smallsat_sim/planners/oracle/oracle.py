from smallsat_sim.planners.base_planner import BasePlanner

import numpy as np


class OraclePlanner(BasePlanner):
    def __init__(
        self,
        env,
        radius=10.5,
        spacing=0.5,
        clearance_dist=0.2,
        plane: str = "yz",
        x_offset: float = 0.0,
        y_offset: float = 0.0,
        z_offset: float = 0.0,
    ) -> None:
        super().__init__(env)
        self.radius = radius
        self.spacing = spacing
        self.clearance_dist = clearance_dist
        self.plane = plane
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.z_offset = z_offset

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

        # Visualize in viewer (if available)
        self.visualize(self.reference_points)

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
        points = np.asarray(self.reference_points)
        point = np.atleast_2d(point)
        if point.shape[1] < 3:
            raise ValueError("point must have at least 3 elements per row.")

        pos = point[:, :3]
        num_points = points.shape[0]

        # Compute distances from pos to each reference point
        distances = np.linalg.norm(
            pos[:, None, :] - points[None, :, :], axis=2
        )  # (N, num_points)
        i_closest = np.argmin(distances, axis=1)

        # Indices for previous and next points (closed loop)
        i_prev = (i_closest - 1) % num_points
        i_next = (i_closest + 1) % num_points

        # Candidate 1: projection onto segment prev -> closest
        A1 = points[i_prev]
        B1 = points[i_closest]
        v1 = B1 - A1
        dot1 = np.sum((pos - A1) * v1, axis=1)
        norm_sq1 = np.sum(v1 * v1, axis=1)
        valid1 = norm_sq1 > 1e-8
        t1 = np.where(valid1, dot1 / norm_sq1, 0.0)
        t1 = np.clip(t1, 0.0, 1.0)
        proj1 = A1 + t1[:, None] * v1
        candidate1 = np.where(valid1[:, None], proj1, B1)

        # Candidate 2: projection onto segment closest -> next
        A2 = points[i_closest]
        B2 = points[i_next]
        v2 = B2 - A2
        dot2 = np.sum((pos - A2) * v2, axis=1)
        norm_sq2 = np.sum(v2 * v2, axis=1)
        valid2 = norm_sq2 > 1e-8
        t2 = np.where(valid2, dot2 / norm_sq2, 0.0)
        t2 = np.clip(t2, 0.0, 1.0)
        proj2 = A2 + t2[:, None] * v2
        candidate2 = np.where(valid2[:, None], proj2, A2)

        # Pick the closer projection for each query
        d1 = np.linalg.norm(pos - candidate1, axis=1)
        d2 = np.linalg.norm(pos - candidate2, axis=1)
        choose_first = d1 <= d2
        best_candidate = np.where(choose_first[:, None], candidate1, candidate2)

        # Arc length approximation based on closest waypoint index
        arc_lengths = i_closest.astype(float) * float(self.spacing)

        if best_candidate.shape[0] == 1:
            return best_candidate[0], arc_lengths[0]
        return best_candidate, arc_lengths

    def _generate_reference(self):
        """
        Generates a circular reference trajectory in the configured plane.
        """
        self.reference_points = []
        if self.radius <= 0:
            self.reference_points.append(
                np.array([self.x_offset, self.y_offset, self.z_offset])
            )
            return

        if self.plane not in {"xy", "yz", "xz"}:
            raise ValueError(f"Unsupported plane '{self.plane}'. Use 'xy', 'yz', or 'xz'.")

        for i in range(int((2 * np.pi * self.radius) // self.spacing)):
            angle = i * self.spacing / self.radius
            if self.plane == "xy":
                x = self.radius * np.cos(angle) + self.x_offset
                y = self.radius * np.sin(angle) + self.y_offset
                z = self.z_offset
            elif self.plane == "yz":
                x = self.x_offset
                y = self.radius * np.sin(angle) + self.y_offset
                z = self.radius * np.cos(angle) + self.z_offset
            else:  # "xz"
                x = self.radius * np.cos(angle) + self.x_offset
                y = self.y_offset
                z = self.radius * np.sin(angle) + self.z_offset

            self.reference_points.append(np.array([x, y, z]))
