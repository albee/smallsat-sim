from smallsat_sim.planners.base_planner import BasePlanner

import numpy as np
import mujoco
import time

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

        # Attitude (convert to quaternion if given as Euler angles in degrees)
        if attitude.shape[0] == 3:
            euler_angles = np.radians(attitude)
            quat = R.from_euler("xyz", euler_angles).as_quat(scalar_first=True)
            self._attitude = quat
        else:
            self._attitude = attitude

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
        velocity: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__(position, attitude, velocity)


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

    @abstractmethod
    def tangent(self, arc_length: Optional[float] = None) -> np.ndarray:
        """
        Method to calculate the tangent at a certain arc_length
        """
        pass

    @abstractmethod
    def closest_point(self, point: np.ndarray) -> tuple[np.ndarray, float]:
        """
        Finds the closest point on the line segment to the given point
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
            # Check if frac_length is out of bound due to numerical errors
            # Otherwise throw an exception
            eps = 1e-3
            if frac_length > 1.0:
                if frac_length < 1.0 + eps:
                    frac_length = 1.0
                else:
                    print(
                        f"Normalized arclength is out of bounds [0,1] with value {frac_length}"
                    )
            elif frac_length < 0.0:
                if frac_length > -eps:
                    frac_length = 0.0
                else:
                    print(
                        f"Normalized arclength is out of bounds [0,1] with value {frac_length}"
                    )

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

    def tangent(self, arc_length: Optional[float] = None) -> np.ndarray:
        direction = self.end_point.position - self.start_point.position

        return direction / np.linalg.norm(direction)

    def closest_point(self, point: np.ndarray) -> tuple[np.ndarray, float]:
        start_to_point = point - self.start_point.position

        tangent = self.tangent()

        projection_length = np.dot(start_to_point, tangent)
        if projection_length <= 0:
            return (self.start_point.position, 0.0)
        elif projection_length >= self.length:
            return (self.end_point.position, self.length)
        else:
            return (
                self.start_point.position + tangent * projection_length,
                projection_length,
            )


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
        self.reference: list[Segment] = []
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
        arc_length %= self.length

        # Identify correct segment to sample from
        segment_index = self._get_segment_index(arc_length)

        # Extract start arc_length
        start_arc_length = self.intervals[segment_index - 1] if segment_index > 0 else 0

        # Retrieve the correct reference from the segment
        segment_arc_length = arc_length - start_arc_length

        return self.reference[segment_index].interpolate(
            segment_arc_length, interpolation_mode="Linear"
        )

    def _get_segment_index(self, arc_length: float) -> int:
        """
        Returns the corresponding segment index wrt. the arc length
        """
        arc_length = arc_length % self.length
        segment_index = 0
        for i, end_arc_length in enumerate(self.intervals):
            if arc_length <= end_arc_length:
                segment_index = i
                break

        return segment_index

    def _get_start_point_segment(self, arc_length: float) -> np.ndarray:
        """
        Returns the starting point of a segment wrt. the arc length
        """
        segment_index = self._get_segment_index(arc_length)

        return self.reference[segment_index].start_point.position

    def _get_start_arc_length_segment(self, arc_length: float) -> np.ndarray:
        """
        Returns the starting point of a segment wrt. the arc length
        """
        segment_index = self._get_segment_index(arc_length)

        if segment_index == 0:
            return np.array([0.0])
        else:
            return np.array([self.intervals[segment_index - 1]])

    def _get_tangent_segment(self, arc_length: float) -> np.ndarray:
        """
        Returns the starting point of a segment wrt. the arc length
        """
        segment_index = self._get_segment_index(arc_length)

        return self.reference[segment_index].tangent(arc_length)


class MissionPlanner(BasePlanner):
    """
    Planning module, which contains a hardcoded trajectory of a
    possible, representative inspection mission around lunar gateway
    Function of the arguments:
        - env: instance of the environment
        - spacing: spacing between intermediate points
        - clearance_dist: clearance distance of waypoints
        - planner_mode: Tracking vs. Path following
            - Waypoint Tracking: WPs are given as references w/o intermediate points
            - Intermediate Waypoint Tracking: WPs are given as reference w/ interme-
                                              mediate points
    """

    def __init__(
        self,
        env,
        spacing=1.0,
        clearance_dist=0.1,
        planner_mode="Intermediate Waypoint Tracking",
    ) -> None:
        super().__init__(env)

        # Initialize paramaters
        self.spacing = spacing
        self.clearance_dist = clearance_dist

        # Load the waypoints
        self._load_waypoints()

        # Create the trajectory
        self._create_trajectory()

        # Set planner mode
        if planner_mode == "Waypoint Tracking":
            # Set the correct get reference method
            self.get_reference = self._get_reference_wp_tracking

            # Initialize current reference point
            self.idx_reference_point = 1

            # Initialize the timer clearance boolean
            self.timer_started = False

        elif planner_mode == "Intermediate Waypoint Tracking":
            # Set the correct get reference method
            self.get_reference = self._get_reference_intermediate_wp_tracking

            # Generate reference with intermediate waypoints
            self._generate_intermediate_reference()

            # Initialize current reference point
            self.idx_reference_point = 1

            # Initialize the timer clearance boolean
            self.timer_started = False

        self._visualize_collision_constraints()

    def _load_waypoints(self) -> None:
        """
        Loads the sparse waypoints that shall be reached
        """
        # Define positional references
        positions = [
            [-3.3, -9, 0],  # Point 1
            [-3.3, -18, 0],  # Point 2
            [-1, -20, 5],  # Point 3
            [16, 0, 23],  # Point 4
            [16, 0, 0],  # Point 5
            [16, 0, -23],  # Point 6
            [16, 0, -26],  # Point 7
            [7, 0, -26],  # Point 8
            [7, 0, -4.5],  # Point 9
            [3, 0, -4.5],  # Point 10
            [3, 0, -8],  # Point 11
            [3.6, 16, -8],  # Point 12
            [3.6, 16, 0],  # Point 13
            [-2.5, 16, 0],  # Point 14
            [-2.5, 5, 0],  # Point 15
            [-7.5, 5, 0],  # Point 16
            [-18, 10, 0],  # Point 17
            [-18, 0, 0],  # Point 18
            [-18, 0, -5],  # Point 19
            [-8, 0, -5],  # Point 20
            [-3.3, 0, -3.5],  # Point 21
            [-3.3, -9, -3.5],  # Point 22
        ]

        # Define attitude references (Euler angles)
        attitudes = [
            [0, 0, 90],  # Point 1
            [0, 0, 90],  # Point 2
            [0, 0, 90],  # Point 3
            [0, 0, 180],  # Point 4
            [0, 0, 180],  # Point 5
            [0, 0, 180],  # Point 6
            [0, 0, 180],  # Point 7
            [0, -90, 0],  # Point 8
            [0, 0, 0],  # Point 9
            [0, -90, 0],  # Point 10
            [0, -180, 0],  # Point 11
            [0, -90, 0],  # Point 12
            [90, 0, -90],  # Point 13
            [0, 0, -90],  # Point 14
            [0, 0, -90],  # Point 15
            [0, 0, -90],  # Point 16
            [0, 0, -45],  # Point 17
            [0, 0, -45],  # Point 18
            [0, -45, 0],  # Point 19
            [0, -90, 0],  # Point 20
            [0, -90, 0],  # Point 21
            [0, -45, 90],  # Point 22
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
            "Line",  # Point 21 to Point 22
            "Line",  # Point 22 to Point 1
        ]

        # Add first point as last point to ensure continuity
        positions.append(positions[0])
        attitudes.append(attitudes[0])

        # Assert that references are setup correctly
        assert (
            len(positions) == len(attitudes) == len(segment_types) + 1
        ), "The number of waypoints must be one more than the number of segment types"

        # Save segment types to self
        self.segment_types = segment_types

        # Create Waypoint objects
        self.waypoints = [
            Waypoint(np.array(pos), np.array(att))
            for pos, att in zip(positions, attitudes)
        ]

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns a reference point based on current observations
        """
        pass

    def _get_reference_wp_tracking(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns the next waypoint as reference.
        Short hold after satellite has reached waypoint.
        """
        # Check if current state is close enough
        dist = np.linalg.norm(
            obs[0:3]
            - self.waypoints[self.idx_reference_point % len(self.waypoints)].position
        )

        # If smallsat enters clearance dist -> start timer
        # Once it's been inside clearance dist for certain time,
        # switch reference to next waypoint
        if dist < self.clearance_dist:
            if not self.timer_started:
                self.timer_started = True
                self.start_time = time.time()
            elif time.time() - self.start_time > 5:
                self.idx_reference_point += 1
                self.timer_started = True

        # Visualize the waypoints and corridor in the saved video
        self._visualize_renderer(
            [waypoint.position for waypoint in self.waypoints],
            size=[0.1, 0, 0],
        )

        return (
            self.waypoints[self.idx_reference_point].position.reshape(3, 1),
            self.waypoints[self.idx_reference_point].attitude.reshape(4, 1),
        )

    def _get_reference_intermediate_wp_tracking(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        # Extract next tracking point
        next_point = self._intermediate_reference[
            self.idx_reference_point % len(self._intermediate_reference)
        ]

        # Check if current state is close enough
        dist = np.linalg.norm(obs[0:3] - next_point.position)

        # If smallsat enters clearance dist -> start timer
        # Once it's been inside clearance dist for certain time,
        # switch reference to next waypoint
        if isinstance(next_point, IntermediateWaypoint):
            if dist < self.clearance_dist:
                self.idx_reference_point += 1
        else:
            if dist < self.clearance_dist:
                if not self.timer_started:
                    self.timer_started = True
                    self.start_time = time.time()
                elif time.time() - self.start_time > 5:
                    self.idx_reference_point += 1
                    self.timer_started = True

        # Visualize the waypoints and corridor in the saved video
        self._visualize_renderer(
            [waypoint.position for waypoint in self._intermediate_reference],
            size=[0.1, 0, 0],
        )

        return (
            self._intermediate_reference[self.idx_reference_point].position.reshape(
                3, 1
            ),
            self._intermediate_reference[self.idx_reference_point].attitude.reshape(
                4, 1
            ),
        )

    def _generate_intermediate_reference(self) -> None:
        """
        Generates a trajectory with intermediate WPs
        """
        self._intermediate_reference = []
        for i, segment in enumerate(self.trajectory.reference):
            # Extract initial arc_length of each interval
            arc_length = self.trajectory.intervals[i - 1] if i > 0 else 0

            # Calculate the number of points in the segment
            num_points_in_segment = int(segment.length / self.spacing)

            # Calculate the actual spacing
            spacing = segment.length / num_points_in_segment

            # Append the Waypoint at the start
            self._intermediate_reference.append(self.waypoints[i])

            # Append Intermediate Waypoints
            self._intermediate_reference.extend(
                self.trajectory.get_intermediate_reference(arc_length + j * spacing)
                for j in range(1, num_points_in_segment)
            )

        if True:
            self.visualize(
                [waypoint.position for waypoint in self._intermediate_reference],
                size=[0.1, 0, 0],
            )

    def _visualize_collision_constraints(self) -> None:
        """
        Visualizes the collision constraints around the mission trajectory
        """

        if hasattr(self, "viewer") and self.viewer is not None:

            collision_radius = 1.0

            for i, segment in enumerate(self.trajectory.reference):
                # Increment ngeom
                self.viewer.user_scn.ngeom += 1

                # Extract points
                start_point = segment.start_point.position
                end_point = segment.end_point.position

                # Initialize geometry
                mujoco.mjv_initGeom(
                    self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.zeros(9),
                    rgba=np.array([0.69, 0.4, 1, 0.1]),
                )

                # Make the connector geometry
                mujoco.mjv_makeConnector(
                    self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    collision_radius,
                    start_point[0],
                    start_point[1],
                    start_point[2],
                    end_point[0],
                    end_point[1],
                    end_point[2],
                )

        else:
            return

    def _create_trajectory(self) -> None:
        """
        Creates trajectory consisting of different connections of waypoints
        """
        self.trajectory = Trajectory(
            waypoints=self.waypoints, segment_types=self.segment_types
        )

    def closest_point_on_trajectory(
        self, point: np.ndarray
    ) -> tuple[np.ndarray, float]:
        """
        Finds the closest point on the trajectory to the given point.
        Returns the position and corresponding arc length.
        """
        closest_point = None
        closest_segment_idx = None
        min_distance = float("inf")

        for i, segment in enumerate(self.trajectory.reference):
            segment_closest_point, segment_arc_length = segment.closest_point(point)
            distance = np.linalg.norm(point - segment_closest_point)
            if distance < min_distance:
                min_distance = distance

                closest_segment_idx = i
                closest_point = segment_closest_point
                closest_absolute_arc_length = segment_arc_length

        # Return coordinates of closest point and arclength
        if closest_segment_idx == 0:
            closest_arc_length = closest_absolute_arc_length
        else:
            closest_arc_length = self.trajectory.intervals[closest_segment_idx - 1]

        return (closest_point, closest_arc_length)
