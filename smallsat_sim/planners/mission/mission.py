from smallsat_sim.planners.base_planner import BasePlanner

import numpy as np
import mujoco

from abc import abstractmethod
from enum import Enum


class Segment:
    """
    Base Class which describes a connection geometry between two waypoints
    """

    def __init__(self, start_point, end_point) -> None:
        self.start_point = start_point
        self.end_point = end_point

        # Evaluate (arc-) length of segment
        self.length = self.calc_length()

    @abstractmethod
    def calc_length(self) -> float:
        pass

    @abstractmethod
    def interpolate(self, arc_length) -> np.ndarray:
        pass


class Line(Segment):
    """
    Connects two waypoints with a line
    """

    def calc_length(self) -> float:
        return np.linalg.norm(self.end_point - self.start_point)

    def interpolate(self, arc_length) -> np.ndarray:
        # Calculate interpolation factor
        frac_length = arc_length / self.length

        # Interpolate coordinates
        interpolated_point = (
            self.start_point + (self.end_point - self.start_point) * frac_length
        )

        return interpolated_point


class Trajectory:
    def __init__(self, waypoints, segment_types: list[str]) -> None:

        # Create the reference
        self._create_reference(waypoints=waypoints, segment_types=segment_types)

        # Calculate the total length of the trajectory
        self.length, self.intervals = self._calc_length_and_intervals()

    def _create_reference(self, waypoints, segment_types: list[str]) -> None:
        self.reference = []
        for i in range(len(waypoints) - 1):
            # Create segment
            segment = self._create_segment(
                segment_type=segment_types[i],
                start_point=waypoints[i],
                end_point=waypoints[i + 1],
            )

            # Add it to the reference
            self.reference.append(segment)

    def _create_segment(
        self, segment_type: str, start_point: np.ndarray, end_point: np.ndarray
    ) -> Segment:
        """
        Creates a segment as part of the trajectory
        """
        if segment_type == "Line":
            return Line(start_point=start_point, end_point=end_point)

    def _calc_length_and_intervals(self) -> tuple[float, list]:
        # Initialize length and intervals
        length = 0
        segments = []

        # Sum over all segments
        for segment in self.reference:
            length += segment.length
            segments.append(length)

        return length, segments

    def get_intermediate_reference(self, arc_length):
        # Modulo with total length (to ensure continuity)
        arc_length = arc_length % self.length

        # Identify correct segment to sample from
        segment_index = 0
        for i, end_arc_length in enumerate(self.intervals):
            if arc_length <= end_arc_length:
                segment_index = i
                break

        # Extract start arc_length
        start_arc_length = 0
        if segment_index != 0:
            start_arc_length = self.intervals[segment_index - 1]

        # Retrieve the correct reference from the segment
        segment_arc_length = arc_length - start_arc_length
        intermediate_waypoint = self.reference[segment_index].interpolate(
            segment_arc_length
        )

        return intermediate_waypoint


class MissionPlanner(BasePlanner):
    def __init__(self, env, spacing=0.5, clearance_dist=0.1) -> None:
        super().__init__(env)

        # Initialize paramaters
        self.spacing = spacing
        self.clearance_dist = clearance_dist

        # Load the waypoints
        self._load_waypoints()

        # Create the trajectory
        self._create_trajectory()

        # Visualize entire trajectory
        if True:
            self._visualize_reference()

    def _load_waypoints(self) -> None:
        """
        Loads the sparse waypoints that shall be reached
        """
        # Sets list of points
        waypoints = [
            [-3.3, -9, 0],  # Point 1
            [-3.3, -25, 0],  # Point 2
            [16, 0, 23],  # Point 3
            [16, 0, 0],  # Point 4
            [16, 0, -23],  # Point 5
            [16, 0, -26],  # Point 6
            [7, 0, -26],  # Point 7
            [7, 0, -4.5],  # Point 8
            [3, 0, -4.5],  # Point 9
        ]

        # Define connection type between points
        segment_types = [
            "Line",  # Point 1 to Point 2
            "Line",  # Point 2 to Point 3
            "Line",  # Point 3 to Point 4
            "Line",  # Point 4 to Point 5
            "Line",  # Point 5 to Point 6
            "Line",  # Point 6 to Point 7
            "Line",  # Point 7 to Point 8
            "Line",  # Point 8 to Point 9
            "Line",  # Point 9 to Point 10
        ]

        # Add first point as last point to ensure continuity
        waypoints.append(waypoints[0])
        segment_types.append(segment_types[0])

        # Convert to numpy for ease of use
        self.waypoints = self._convert_to_numpy(waypoints)
        self.segment_types = segment_types

        # Visualize wapoints for DEBUG
        if True:
            self.visualize(self.waypoints, size=[0.2, 0, 0])

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns a reference point based on current observations
        """
        return np.array([0, 0, 10])

    def _visualize_reference(self):
        """
        Visualize the reference as continuous trajectory
        """
        # Retrieve total length of trajectory
        tot_len = self.trajectory.length

        # Desired spacing
        N_points = 100
        spacing = tot_len / N_points

        offset = self.viewer.user_scn.ngeom
        for i in range(N_points):
            point = self.trajectory.get_intermediate_reference(i * spacing)
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i + offset],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.1, 0, 0],
                pos=point,
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 2]),
            )

        self.viewer.user_scn.ngeom += N_points

    def _create_trajectory(self) -> None:
        """
        Creates trajectory consisting of different connections of waypoints
        """
        self.trajectory = Trajectory(
            waypoints=self.waypoints, segment_types=self.segment_types
        )

    def _convert_to_numpy(self, list: list) -> list:
        """
        Converts all elements inside list to numpy arrays
        """
        # Initialize new list
        converted_list = []

        # Conversion of each element
        for item in list:
            converted_item = np.array(item)
            converted_list.append(converted_item)

        return converted_list
