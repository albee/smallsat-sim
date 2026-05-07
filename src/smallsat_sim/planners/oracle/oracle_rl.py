from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.envs.vec_env import VecEnv

import jax.numpy as jnp


class OraclePlannerRL(BasePlanner):
    def __init__(
        self, env: VecEnv, radius=3.0, spacing=1.0, clearance_dist=0.2
    ) -> None:
        super().__init__(env)
        self.radius = radius
        self.spacing = spacing
        self.clearance_dist = jnp.full(env.num_envs, clearance_dist)

        # Initialize current reference point
        self.reference_point_indices = jnp.zeros(env.num_envs, dtype=jnp.int32)

        # Flags to indicate whether the agents have completed the path
        self.completed_path = jnp.zeros(env.num_envs, dtype=jnp.bool_)

        # Generate a circular reference trajectory around the gateway
        self._generate_reference()

        # Visualize the circle
        self.visualize(self.reference_point_list)

    def get_reference(self, obs: jnp.ndarray) -> jnp.ndarray:
        """
        Get the next waypoint on the reference trajectory (not the attitude), for each environment.
        """
        num_points = self.reference_points.shape[0]
        # Get the current reference point for each environment
        ref_points = self.reference_points[self.reference_point_indices % num_points]
        # Compute the distance for each environment
        dist = jnp.linalg.norm(obs[:, 0:3] - ref_points, axis=1)

        # If smallsat is closer than the clearance distance, advance to the next reference point
        self.reference_point_indices = self.reference_point_indices + jnp.where(
            dist < self.clearance_dist, 1, 0
        )

        # Check if the agents have completed the path
        self.completed_path = jnp.where(
            jnp.logical_or(
                self.reference_point_indices >= num_points, self.completed_path
            ),
            True,
            False,
        )

        # Visualize in viewer (if available)
        self.visualize(self.reference_point_list)

        # Visualize the waypoints in the saved video
        self._visualize_renderer(self.reference_point_list)

        ref_pos = self.reference_points[self.reference_point_indices % num_points]
        ref_quat = jnp.tile(jnp.array([1.0, 0.0, 0.0, 0.0]), (obs.shape[0], 1))

        return jnp.concatenate([ref_pos, ref_quat], axis=1)

    def closest_point_on_trajectory(self, point: jnp.ndarray) -> jnp.ndarray:
        """
        Finds the orthogonal projection of the smallsat position onto the trajectory.
        """
        # Extract the smallsat's current positions (assumed to be the first three elements)
        pos = point[:, :3]  # shape: (num_envs, 3)

        # Compute distances from pos to each reference point
        distances = jnp.linalg.norm(
            pos[:, None, :] - self.reference_points[None, :, :], axis=2
        )  # (num_envs, num_reference_points)

        # For each environment, find the index of the closest reference point
        i_closest = jnp.argmin(distances, axis=1)  # shape: (num_envs,)
        num_points = self.reference_points.shape[0]

        # Compute indices for the previous and next points (with wrap-around for closed loop)
        i_prev = (i_closest - 1) % num_points
        i_next = (i_closest + 1) % num_points

        # Candidate 1: projection on the segment from the previous point to the closest point
        A1 = self.reference_points[i_prev]  # shape: (num_envs, 3)
        B1 = self.reference_points[i_closest]  # shape: (num_envs, 3)
        v1 = B1 - A1  # shape: (num_envs, 3)
        dot1 = jnp.sum((pos - A1) * v1, axis=1)  # shape: (num_envs,)
        norm_sq1 = jnp.sum(v1 * v1, axis=1)  # shape: (num_envs,)
        valid1 = norm_sq1 > 1e-8
        t1 = jnp.where(valid1, dot1 / norm_sq1, 0.0)
        t1 = jnp.clip(t1, 0.0, 1.0)  # ensure projection lies on the segment
        proj1 = A1 + t1[:, None] * v1  # shape: (num_envs, 3)
        candidate1 = jnp.where(valid1[:, None], proj1, B1)

        # Candidate 2: projection on the segment from the closest point to the next point
        A2 = self.reference_points[i_closest]  # shape: (num_envs, 3)
        B2 = self.reference_points[i_next]  # shape: (num_envs, 3)
        v2 = B2 - A2  # shape: (num_envs, 3)
        dot2 = jnp.sum((pos - A2) * v2, axis=1)  # shape: (num_envs,)
        norm_sq2 = jnp.sum(v2 * v2, axis=1)  # shape: (num_envs,)
        valid2 = norm_sq2 > 1e-8
        t2 = jnp.where(valid2, dot2 / norm_sq2, 0.0)
        t2 = jnp.clip(t2, 0.0, 1.0)
        proj2 = A2 + t2[:, None] * v2  # shape: (num_envs, 3)
        candidate2 = jnp.where(valid2[:, None], proj2, A2)

        # Compute distances from pos to each candidate projection
        d1 = jnp.linalg.norm(pos - candidate1, axis=1)
        d2 = jnp.linalg.norm(pos - candidate2, axis=1)

        # Choose the candidate with the smaller distance for each environment
        choose_candidate1 = d1 <= d2  # boolean mask, shape: (num_envs,)
        best_candidate = jnp.where(choose_candidate1[:, None], candidate1, candidate2)

        return best_candidate

    def _generate_reference(self):
        """
        Generates a circle in the yz plane around the gateway
        """
        if self.radius == 0.0:  # Regulate to the origin
            self.reference_point_list = [jnp.array([0, 0, 10.17])]
            self.reference_points = jnp.array([[0, 0, 10.17]])

        else:  # Draw a circle
            self.reference_point_list = []
            for i in range(int((2 * jnp.pi * self.radius) // self.spacing)):
                x = self.radius * jnp.cos(i * self.spacing / self.radius)
                y = self.radius * jnp.sin(i * self.spacing / self.radius)
                z = 10.17  # z-coordinate remains constant as the circle is in the xy-plane
                self.reference_point_list.append(jnp.array([x, y, z]))

            num_points = int((2 * jnp.pi * self.radius) // self.spacing)
            if num_points < 1:
                raise ValueError("Radius too small for the given spacing.")

            indices = jnp.arange(num_points)
            x = self.radius * jnp.cos(indices * self.spacing / self.radius)
            y = self.radius * jnp.sin(indices * self.spacing / self.radius)
            z = jnp.full_like(x, 10.17)  # constant z-coordinate
            self.reference_points = jnp.stack([x, y, z], axis=1)
