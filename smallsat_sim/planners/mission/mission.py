from smallsat_sim.planners.base_planner import BasePlanner

import numpy as np
import mujoco

from abc import abstractmethod
from typing import Optional
from scipy.spatial.transform import Slerp, Rotation as R


class Waypoint:
    """
    Waypoint class which encodes the following references at a setpoint:
        - Position reference
        - Attitude reference
        - [Optional] Velocity reference
    """

    def __init__(
        self,
        position: np.ndarray,
        attitude: np.ndarray,
        velocity: Optional[np.ndarray] = None,
    ) -> None:
        # Position in intertial frame
        self._position = position

        # Attitude (convert to quaternion from Euler angles in degrees)
        euler_angles = np.radians(attitude)
        quat = R.from_euler("xyz", euler_angles).as_quat(scalar_first=True)
        self._attitude = quat

        # Velocity in body frame
        self._velocity = velocity

    @property
    def position(self) -> np.ndarray:
        """
        Getter method for the position
        """
        return self._position

    @property
    def attitude(self) -> np.ndarray:
        """
        Getter method for the attitude
        """
        return self._attitude

    @property
    def velocity(self) -> np.ndarray:
        """
        Getter method for the velocity
        """
        return self._velocity


class IntermediateWaypoint(Waypoint):
    """
    Intermediate Waypoint class which inherits from Waypoint
    """

    def __init__(
        self,
        position: np.ndarray,
        attitude: np.ndarray,
        velocity: np.ndarray | None = None,
    ) -> None:
        # Position in intertial frame
        self._position = position

        # Attitude (already given as quaternion)
        self._attitude = attitude

        # Velocity in body frame
        self._velocity = velocity


class Segment:
    """
    Base Class which describes a connection geometry between two waypoints
    """

    def __init__(self, start_point: Waypoint, end_point: Waypoint) -> None:
        self.start_point = start_point
        self.end_point = end_point

        # Evaluate (arc-) length of segment
        self.length = self.calc_length()

    @abstractmethod
    def calc_length(self) -> float:
        """
        Method for calculating the length of a segment
        """
        pass

    @abstractmethod
    def interpolate(
        self, arc_length: float, interpolation_mode: str
    ) -> IntermediateWaypoint:
        """
        Method to calculate an intermediate waypoint.
            - arc_length: Arc length of intermediate waypoint to calculate
            - interpolation_mode: Method of how to calculate intermediate
              reference for attitude (or velocity). Possibilites:
                - Constant
                - Linear interpolation
        """
        pass


class Line(Segment):
    """
    Connects two waypoints with a line
    """

    def calc_length(self) -> float:
        return np.linalg.norm(self.end_point.position - self.start_point.position)

    def interpolate(
        self, arc_length: float, interpolation_mode: str = "Linear"
    ) -> np.ndarray:
        # Calculate interpolation factor
        frac_length = arc_length / self.length

        # Interpolate position
        interpolated_position = (
            self.start_point.position
            + (self.end_point.position - self.start_point.position) * frac_length
        )

        # Interpolate attitude
        if interpolation_mode == "Linear":
            # Create slerp object
            slerp = Slerp(
                times=[0, 1],
                rotations=R.from_quat(
                    [self.start_point.attitude, self.end_point.attitude],
                    scalar_first=True,
                ),
            )
            interpolated_attitude = slerp(times=frac_length).as_quat(scalar_first=True)
        elif interpolation_mode == "Constant":
            interpolated_attitude = self.end_point.attitude
        else:
            raise ValueError(
                f"Interpolation mode '{interpolation_mode}' is not defined."
            )

        # Create Waypoint
        intermediate_waypoint = IntermediateWaypoint(
            position=interpolated_position, attitude=interpolated_attitude
        )

        return intermediate_waypoint


class Trajectory:
    """
    This class holds all segments making up the entire trajectory
    """

    def __init__(self, waypoints: list[Waypoint], segment_types: list[str]) -> None:

        # Create the reference
        self._create_reference(waypoints=waypoints, segment_types=segment_types)

        # Calculate the total length of the trajectory
        self.length, self.intervals = self._calc_length_and_intervals()

    def _create_reference(
        self, waypoints: list[Waypoint], segment_types: list[str]
    ) -> None:
        """
        Creates the reference by creating a list of segments
        """
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
        self, segment_type: str, start_point: Waypoint, end_point: Waypoint
    ) -> Segment:
        """
        Creates a segment as part of the trajectory
        """
        if segment_type == "Line":
            return Line(start_point=start_point, end_point=end_point)
        else:
            raise ValueError(f"Unsupported segment type: {segment_type}")

    def _calc_length_and_intervals(self) -> tuple[float, list[float]]:
        """
        Calculates the following quantities:
            - Total length of trajectory
            - Length intervals of segments
        """
        # Initialize length and intervals
        length = 0
        segments = []

        # Sum over all segments
        for segment in self.reference:
            length += segment.length
            segments.append(length)

        return length, segments

    def get_intermediate_reference(self, arc_length: float) -> IntermediateWaypoint:
        """
        Retrieves the correct reference wrt. to the given arc length
        """
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
    """
    Planning module, which contains a hardcoded trajectory of a
    possible, representative inspection mission around lunar gateway
    """

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
        # Define positional references
        positions = [
            [-3.3, -9, 0],  # Point 1
            [-3.3, -25, 0],  # Point 2
            [16, 0, 23],  # Point 3
            [16, 0, 0],  # Point 4
            [16, 0, -23],  # Point 5
            [16, 0, -26],  # Point 6
            [7, 0, -26],  # Point 7
            [7, 0, -4.5],  # Point 8
            [3, 0, -4.5],  # Point 9
            [3, 0, -8],  # Point 10
            [3.6, 16, -8],  # Point 11
            [3.6, 16, 0],  # Point 12
            [-2.5, 16, 0],  # Point 13
            [-2.5, 5, 0],  # Point 14
            [-7.5, 5, 0],  # Point 15
            [-18, 10, 0],  # Point 16
            [-18, 0, 0],  # Point 17
            [-18, 0, -5],  # Point 18
            [-8, 0, -5],  # Point 19
            [-3.3, 0, -3.5],  # Point 20
            [-3.3, -9, -3.5],  # Point 21
        ]

        # Define attitude references (Euler angles)
        attitudes = [
            [0, 0, 0],  # Point 1
            [0, 0, 0],  # Point 2
            [0, 0, 0],  # Point 3
            [0, 0, 0],  # Point 4
            [0, 0, 0],  # Point 5
            [0, 0, 0],  # Point 6
            [0, 0, 0],  # Point 7
            [0, 0, 0],  # Point 8
            [0, 0, 0],  # Point 9
            [0, 0, 0],  # Point 10
            [0, 0, 0],  # Point 11
            [0, 0, 0],  # Point 12
            [0, 0, 0],  # Point 13
            [0, 0, 0],  # Point 14
            [0, 0, 0],  # Point 15
            [0, 0, 0],  # Point 16
            [0, 0, 0],  # Point 17
            [0, 0, 0],  # Point 18
            [0, 0, 0],  # Point 19
            [0, 0, 0],  # Point 20
            [0, 0, 0],  # Point 21
        ]

        # Define connection type between waypoints
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
            "Line",  # Point 10 to Point 11
            "Line",  # Point 11 to Point 12
            "Line",  # Point 12 to Point 13
            "Line",  # Point 13 to Point 14
            "Line",  # Point 14 to Point 15
            "Line",  # Point 15 to Point 16
            "Line",  # Point 16 to Point 17
            "Line",  # Point 17 to Point 18
            "Line",  # Point 18 to Point 19
            "Line",  # Point 19 to Point 20
            "Line",  # Point 20 to Point 21
            "Line",  # Point 21 to Point 1
        ]

        # Add first point as last point to ensure continuity
        positions.append(positions[0])
        attitudes.append(attitudes[0])

        # Assert that references are setup correctly
        assert (
            len(positions) == len(attitudes) == len(segment_types) + 1
        ), "The number of waypoints must be one more than the number of segment types"

        # Convert to numpy for ease of use
        self.positions = self._convert_to_numpy(positions)
        self.attitudes = self._convert_to_numpy(attitudes)
        self.segment_types = segment_types

        # Create Waypoint objects
        self.waypoints = []
        for i in range(len(positions)):
            self.waypoints.append(
                Waypoint(
                    position=self.positions[i],
                    attitude=self.attitudes[i],
                )
            )

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns a reference point based on current observations
        """
        return np.array([0, 0, 10])

    def _visualize_reference(self) -> None:
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
            point = self.trajectory.get_intermediate_reference(i * spacing).position
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

    def _convert_to_numpy(self, list: list[list[float]]) -> list[np.ndarray]:
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
