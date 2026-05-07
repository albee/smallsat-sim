from __future__ import annotations

import numpy as np

from smallsat_sim.planners.base_planner import BasePlanner


class DockingPlanner(BasePlanner):
    """
    Two-stage docking planner:
    1) Fly to pre-dock point
    2) Fly to final dock point
    """

    def __init__(
        self,
        env,
        pre_dock_position: np.ndarray,
        dock_position: np.ndarray,
        dock_attitude: np.ndarray | None = None,
        switch_distance: float = 0.35,
        dock_distance: float = 0.15,
    ) -> None:
        super().__init__(env)

        self.pre_dock_position = np.asarray(pre_dock_position, dtype=float).reshape(3)
        self.dock_position = np.asarray(dock_position, dtype=float).reshape(3)

        if dock_attitude is None:
            dock_attitude = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        self.dock_attitude = np.asarray(dock_attitude, dtype=float).reshape(4)

        self.switch_distance = float(switch_distance)
        self.dock_distance = float(dock_distance)
        self.stage = 0

        # MPC controllers manage renderer visualization; avoid duplicate frames.
        self.is_mpc = True

    def get_reference(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        position = np.asarray(obs[:3], dtype=float).reshape(3)

        if self.stage == 0:
            dist_to_pre_dock = np.linalg.norm(position - self.pre_dock_position)
            if dist_to_pre_dock <= self.switch_distance:
                self.stage = 1

        target_position = (
            self.pre_dock_position if self.stage == 0 else self.dock_position
        )

        points = [self.pre_dock_position, self.dock_position]
        self.visualize(points, size=[0.03, 0, 0])
        self._visualize_renderer(points, size=[0.03, 0, 0])

        return target_position.reshape(3, 1), self.dock_attitude.reshape(4, 1)

    def closest_point_on_trajectory(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, float]:
        point = np.asarray(point, dtype=float).reshape(3)
        start = self.pre_dock_position
        end = self.dock_position

        segment = end - start
        segment_length = np.linalg.norm(segment)
        if segment_length < 1e-12:
            return start.copy(), 0.0

        t = np.dot(point - start, segment) / np.dot(segment, segment)
        t_clipped = float(np.clip(t, 0.0, 1.0))
        closest = start + t_clipped * segment
        arc_length = t_clipped * segment_length

        return closest, arc_length

    def is_docked(self, obs: np.ndarray) -> bool:
        position = np.asarray(obs[:3], dtype=float).reshape(3)
        return np.linalg.norm(position - self.dock_position) <= self.dock_distance
