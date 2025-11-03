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

        if direction.all() == 0:
            return [0, 0, 0]

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
        Returns the tangent of a segment wrt. the arc length
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
        spacing=0.2,
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

        # Capture current position to align the initial reference with the vehicle state
        initial_obs = None
        if hasattr(env, "get_obs"):
            try:
                initial_obs = env.get_obs()
            except TypeError:
                initial_obs = None
        if initial_obs is None:
            initial_obs = getattr(env, "obs", None)
        if initial_obs is not None:
            initial_position = np.asarray(initial_obs[:3]).reshape(3)
        else:
            initial_position = self.waypoints[0].position.copy()

        # Set planner mode
        if planner_mode == "Waypoint Tracking":
            # Set the correct get reference method
            self.get_reference = self._get_reference_wp_tracking

            # Initialize current reference point
            self.idx_reference_point = self._closest_waypoint_index(initial_position)

            # Initialize the timer clearance boolean
            self.timer_started = False

        elif planner_mode == "Intermediate Waypoint Tracking":
            # Set the correct get reference method
            self.get_reference = self._get_reference_intermediate_wp_tracking

            # Generate reference with intermediate waypoints
            self._generate_intermediate_reference()

            # Initialize current reference point
            self.idx_reference_point = self._closest_intermediate_index(
                initial_position
            )

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
            [-1.5, 10.5, -2],  # Point 15
            [-2.0, 7.75, -1],  # Point 16
            [-5.0, 5.0, 0],  # Point 17
            [-7.5, 5, 0],  # Point 18
            [-18, 10, 0],  # Point 19
            [-18, 0, 0],  # Point 20
            [-18, 0, -5],  # Point 21
            [-8, 0, -5],  # Point 22
            [-3.3, 0, -3.5],  # Point 23
            [-3.3, -9, -3.5],  # Point 24
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
            [0, 0, -90],  # Point 17
            [0, 0, -90],  # Point 18
            [0, 0, -45],  # Point 19
            [0, 0, -45],  # Point 20
            [0, -45, 0],  # Point 21
            [0, -90, 0],  # Point 22
            [0, -90, 0],  # Point 23
            [0, -45, 90],  # Point 24
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
            "Line",  # Point 14 to Point 15
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

    def _closest_waypoint_index(self, position: np.ndarray) -> int:
        distances = [
            np.linalg.norm(position - waypoint.position) for waypoint in self.waypoints
        ]
        return int(np.argmin(distances))

    def _closest_intermediate_index(self, position: np.ndarray) -> int:
        if (
            not hasattr(self, "_intermediate_reference")
            or not self._intermediate_reference
        ):
            return self._closest_waypoint_index(position)

        distances = [
            np.linalg.norm(position - waypoint.position)
            for waypoint in self._intermediate_reference
        ]
        return int(np.argmin(distances))

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns a reference point based on current observations
        """
        return self._get_reference_wp_tracking(obs)

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
                self.timer_started = False

        # Visualize in viewer (if available)
        self.visualize(
            [waypoint.position for waypoint in self.waypoints],
            size=[0.05, 0, 0],
        )

        # Visualize the waypoints and corridor in the saved video
        self._visualize_renderer(
            [waypoint.position for waypoint in self.waypoints],
            size=[0.05, 0, 0],
        )

        reference = self.waypoints[self.idx_reference_point % len(self.waypoints)]

        return (
            reference.position.reshape(3, 1),
            reference.attitude.reshape(4, 1),
        )

    def _get_reference_intermediate_wp_tracking(
        self, obs: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:

        # Extract next tracking point
        next_point_idx = self.idx_reference_point % len(self._intermediate_reference)
        next_point = self._intermediate_reference[next_point_idx]

        dist = np.linalg.norm(obs[0:3] - next_point.position)

        # Check if current state is close enough
        if dist < self.clearance_dist:
            self.idx_reference_point += 1

        # Visualize in viewer (if available)
        self.visualize(
            [waypoint.position for waypoint in self._intermediate_reference],
            size=[0.05, 0, 0],
        )

        # Visualize the waypoints and corridor in the saved video
        self._visualize_renderer(
            [waypoint.position for waypoint in self._intermediate_reference],
            size=[0.05, 0, 0],
        )

        reference = self._intermediate_reference[
            self.idx_reference_point % len(self._intermediate_reference)
        ]

        return (
            reference.position.reshape(3, 1),
            reference.attitude.reshape(4, 1),
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
            spacing = segment.length / (1e-5 + num_points_in_segment)

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
                size=[0.05, 0, 0],
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
                    rgba=np.array([0.69, 0.4, 1, 0.0]),
                )

                # Make the connector geometry
                mujoco.mjv_connector(
                    self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    float(collision_radius),
                    [
                        float(start_point[0]),
                        float(start_point[1]),
                        float(start_point[2]),
                    ],
                    [float(end_point[0]), float(end_point[1]), float(end_point[2])],
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
            closest_arc_length = (
                closest_absolute_arc_length
                + self.trajectory.intervals[closest_segment_idx - 1]
            )

        return (closest_point, closest_arc_length)

    def distance_to_closest_waypoint(self, point: np.ndarray) -> float:
        """
        Calculates the distance from the given point to the closest Waypoint.
        Returns the distance and the index of the closest Waypoint.
        """
        min_distance = float("inf")
        closest_waypoint_index = -1

        for idx, waypoint in enumerate(self.waypoints):
            distance = np.linalg.norm(point - waypoint.position)
            if distance < min_distance:
                min_distance = distance
                closest_waypoint_index = idx

        return min_distance

    def _visualize_renderer(
        self, points: list[np.ndarray], color=[1, 0, 0, 2], size=[0.05, 0, 0]
    ) -> None:
        """
        Visualizes reference points in the MuJoCo renderer.
        """
        if (
            self.data.time >= self.env_cfg.renderer.start_recording
            and self.data.time <= self.env_cfg.renderer.end_recording
        ):
            self.renderer.scene.ngeom = 0

            # Update the renderer scene
            self.renderer.update_scene(self.data, self.cam)

            # # Iterate over all points which need to be visualized in renderer
            for point in points:
                self.renderer.scene.ngeom += 1
                x = point[0]
                y = point[1]
                z = point[2]
                mujoco.mjv_initGeom(
                    self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=size,
                    pos=np.array([x, y, z]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(color),
                )

            # Collision constraints
            collision_radius = 1.0
            for i, segment in enumerate(self.trajectory.reference):
                # Increment ngeom
                self.renderer.scene.ngeom += 1

                # Extract points
                start_point = segment.start_point.position
                end_point = segment.end_point.position

                # Initialize geometry
                mujoco.mjv_initGeom(
                    self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                    size=np.zeros(3),
                    pos=np.zeros(3),
                    mat=np.zeros(9),
                    rgba=np.array([0.69, 0.4, 1, 0.1]),
                )

                # Make the connector geometry
                mujoco.mjv_connector(
                    self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
                    mujoco.mjtGeom.mjGEOM_CAPSULE,
                    float(collision_radius),
                    [
                        float(start_point[0]),
                        float(start_point[1]),
                        float(start_point[2]),
                    ],
                    [float(end_point[0]), float(end_point[1]), float(end_point[2])],
                )

            # Extract image from renderer and append it for post-processing
            if not self.is_mpc:
                sim_img = self.renderer.render().copy()
                self.frames.append(sim_img)


"""
Cubic splines implementation
"""
import numpy as np
from scipy.interpolate import CubicSpline


class MissionPlannerCubicSpline(MissionPlanner):
    """
    Planning module which contains a hardcoded trajectory represented by a cubic spline
    """

    def __init__(
        self,
        env,
        spacing=1.0,
    ) -> None:

        # Initialize parameters
        self.spacing = spacing

        # Load the waypoints
        self._load_waypoints_CS()

        # Create the trajectory
        self._create_trajectory_CS()

        # Generate visualization waypoints
        self.visualization_points = self.get_visualization_points(spacing=0.1)

        super().__init__(env)

    def _load_waypoints_CS(self) -> None:
        """
        Loads the sparse waypoints that shall be reached
        """
        # Define positional references
        self.positions = [
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
            [-1.5, 12, -2],  # Point 15
            [-2.0, 6.0, 0],  # Point 16
            [-4.0, 5.5, 0],  # Point 17
            [-10, 5, 0],  # Point 18
            [-18, 10, 0],  # Point 19
            [-18, 0, 0],  # Point 20
            [-18, 0, -5],  # Point 21
            [-8, 0, -5],  # Point 22
            [-3.3, 0, -3.5],  # Point 23
            [-3.3, -9, -3.5],  # Point 24
        ]

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
            [0, 0, -90],  # Point 17
            [0, 0, -90],  # Point 18
            [0, 0, -45],  # Point 19
            [0, 0, -45],  # Point 20
            [0, -45, 0],  # Point 21
            [0, -90, 0],  # Point 22
            [0, -90, 0],  # Point 23
            [0, -45, 90],  # Point 24
        ]

        # Convert attitudes from Euler angles (degrees) to quaternions
        # Assuming Euler angles are in ZYX order and scalar_first=True
        self.attitudes = R.from_euler("ZYX", attitudes, degrees=True).as_quat()
        # Adjust to scalar_first format [w, x, y, z]
        self.attitudes = self.attitudes[:, [3, 0, 1, 2]]  # Reorder to [w, x, y, z]

    def _create_trajectory_CS(self) -> None:
        """
        Creates a continuous trajectory made up of cubic splines interpolating between the waypoints.
        Interpolates both positions and attitudes.
        """
        # Convert waypoints to numpy array
        waypoints = np.array(self.positions)
        attitudes = self.attitudes  # Quaternions [w, x, y, z]
        num_waypoints = len(waypoints)
        # Initialize lists to hold interpolated points and attitudes
        interpolated_points = []
        interpolated_attitudes = []
        cumulative_lengths = [0.0]  # Initialize cumulative arc lengths

        # Loop over each segment between waypoints, including the last segment back to the first waypoint
        for i in range(num_waypoints):
            start_point = waypoints[i]
            end_point = waypoints[
                (i + 1) % num_waypoints
            ]  # Wrap around to the first waypoint
            start_attitude = attitudes[i]
            end_attitude = attitudes[(i + 1) % num_waypoints]

            # Calculate distance between points
            segment_length = np.linalg.norm(end_point - start_point)

            # Skip zero-length segments
            if segment_length == 0:
                continue

            # Determine number of points needed (at least 1 to include start and end)
            num_points = max(int(np.floor(segment_length / self.spacing)), 1)

            # Generate linearly spaced interpolation parameters
            t_values = np.linspace(0, 1, num_points + 1)  # Include start and end

            # Reorder quaternions to [x, y, z, w] format for scipy Rotation
            start_attitude_xyzw = start_attitude[[1, 2, 3, 0]]  # [x, y, z, w]
            end_attitude_xyzw = end_attitude[[1, 2, 3, 0]]  # [x, y, z, w]

            # Create rotations for slerp
            key_times = [0, 1]
            key_rots = R.from_quat([start_attitude_xyzw, end_attitude_xyzw])

            slerp = Slerp(key_times, key_rots)

            # Interpolate positions and attitudes
            for t in t_values:
                # Interpolate position
                point = (1 - t) * start_point + t * end_point
                # Avoid adding duplicate points
                if len(interpolated_points) > 0 and np.allclose(
                    point, interpolated_points[-1]
                ):
                    continue
                interpolated_points.append(point)
                # Interpolate attitude using slerp
                interp_rot = slerp([t])[0]
                interp_quat_xyzw = interp_rot.as_quat()
                # Convert back to [w, x, y, z] format
                interp_quat_wxyz = interp_quat_xyzw[[3, 0, 1, 2]]
                interpolated_attitudes.append(interp_quat_wxyz)

                # Update cumulative arc lengths
                if len(interpolated_points) > 1:
                    delta_length = np.linalg.norm(
                        interpolated_points[-1] - interpolated_points[-2]
                    )
                    cumulative_lengths.append(cumulative_lengths[-1] + delta_length)

        # Ensure periodicity by checking if the first and last points are the same
        if not np.allclose(interpolated_points[0], interpolated_points[-1]):
            # Append the first point and attitude to ensure periodicity
            interpolated_points.append(interpolated_points[0])
            interpolated_attitudes.append(interpolated_attitudes[0])
            # Update cumulative length
            delta_length = np.linalg.norm(
                interpolated_points[-1] - interpolated_points[-2]
            )
            cumulative_lengths.append(cumulative_lengths[-1] + delta_length)

        # Convert to numpy arrays
        interpolated_points = np.array(interpolated_points)
        interpolated_attitudes = np.array(interpolated_attitudes)
        s = np.array(cumulative_lengths)

        # Remove any duplicate s values to ensure s is strictly increasing
        s_diff = np.diff(s)
        if not np.all(s_diff > 0):
            # Find indices where s increases
            increasing_indices = (
                np.where(s_diff > 0)[0] + 1
            )  # indices where s increases
            valid_indices = np.concatenate(([0], increasing_indices))
            s = s[valid_indices]
            interpolated_points = interpolated_points[valid_indices]
            interpolated_attitudes = interpolated_attitudes[valid_indices]

        # Store total length for modulo operation
        self.total_length = s[-1]

        # Fit cubic splines x(s), y(s), z(s)
        self.x_spline = CubicSpline(s, interpolated_points[:, 0], bc_type="periodic")
        self.y_spline = CubicSpline(s, interpolated_points[:, 1], bc_type="periodic")
        self.z_spline = CubicSpline(s, interpolated_points[:, 2], bc_type="periodic")

        # Store the interpolated attitudes and cumulative lengths for attitude interpolation
        self.interpolated_attitudes = interpolated_attitudes
        self.s_attitudes = s

    def get_attitude(self, s):
        """
        Given an arc length s, return the interpolated quaternion at that point.
        """
        # Apply modulo operation to handle wrapping around the trajectory
        s = s % self.total_length

        # Find the segment index corresponding to s
        idx = np.searchsorted(self.s_attitudes, s) - 1
        idx = np.clip(idx, 0, len(self.s_attitudes) - 2)

        # Get the start and end attitudes and their arc lengths
        s0 = self.s_attitudes[idx]
        s1 = self.s_attitudes[idx + 1]
        t = (s - s0) / (s1 - s0) if s1 > s0 else 0.0  # Avoid division by zero

        # Reorder quaternions to [x, y, z, w] format
        attitude1_xyzw = self.interpolated_attitudes[idx][[1, 2, 3, 0]]
        attitude2_xyzw = self.interpolated_attitudes[idx + 1][[1, 2, 3, 0]]

        # Create rotations for slerp
        key_times = [s0, s1]
        key_rots = R.from_quat([attitude1_xyzw, attitude2_xyzw])

        slerp = Slerp(key_times, key_rots)
        interp_rot = slerp([s])[0]
        interp_quat_xyzw = interp_rot.as_quat()
        # Convert back to [w, x, y, z] format
        interp_quat_wxyz = interp_quat_xyzw[[3, 0, 1, 2]]
        return interp_quat_wxyz

    def get_spline_parameters(self, s):
        """
        Given an arc length s, return the coefficients of the cubic spline segment that contains s.
        """
        # Apply modulo operation to handle wrapping around the trajectory
        s = s % self.total_length
        # Find the segment index corresponding to s
        idx = np.searchsorted(self.x_spline.x, s, side="right") - 1
        idx = np.clip(idx, 0, len(self.x_spline.x) - 2)
        # Extract coefficients for x, y, z
        x_coeffs = self.x_spline.c[:, idx]
        y_coeffs = self.y_spline.c[:, idx]
        z_coeffs = self.z_spline.c[:, idx]
        # The spline segment is defined over [s0, s1]
        s0 = self.x_spline.x[idx]
        s1 = self.x_spline.x[idx + 1]
        return {
            "segment_index": idx,
            "s_range": (s0, s1),
            "x_coeffs": x_coeffs,
            "y_coeffs": y_coeffs,
            "z_coeffs": z_coeffs,
        }

    def get_visualization_points(self, spacing=0.1) -> list[IntermediateWaypoint]:
        """
        Generates IntermediateWaypoint objects along the trajectory at specified intervals for visualization purposes.
        Returns a list of IntermediateWaypoint objects.
        """
        # Total arc length
        total_length = self.total_length
        # Number of samples, ensure we include the last point
        num_samples = int(np.ceil(total_length / spacing)) + 1
        # Generate arc lengths at specified spacing
        s_values = np.linspace(0, total_length, num_samples)

        # Evaluate spline functions at these arc lengths
        x_vals = self.x_spline(s_values)
        y_vals = self.y_spline(s_values)
        z_vals = self.z_spline(s_values)

        # Initialize list of IntermediateWaypoint objects
        waypoints = []

        # Interpolate attitudes at these arc lengths
        attitudes = []
        for s in s_values:
            quat = self.get_attitude(s)
            attitudes.append(quat)

        # Create IntermediateWaypoint objects
        velocity = np.array([0.0, 0.0, 0.0])  # Zero velocity

        for x, y, z, quat in zip(x_vals, y_vals, z_vals, attitudes):
            position = np.array([x, y, z])
            attitude = quat  # Quaternion [w, x, y, z]
            waypoint = IntermediateWaypoint(position, attitude, velocity)
            waypoints.append(waypoint)

        return waypoints

    def closest_point_on_trajectory_CS(self, point):
        """
        Given a 3D point, find the closest point on the spline and the corresponding arc length.
        Returns:
            closest_point (np.ndarray): The closest point on the spline.
            s_closest (float): The arc length corresponding to the closest point.
        """
        from scipy.optimize import minimize_scalar

        # Objective function: squared distance between spline point at s and the given point
        def objective(s):
            s = s % self.total_length  # Wrap around the total length
            x = self.x_spline(s)
            y = self.y_spline(s)
            z = self.z_spline(s)
            spline_point = np.array([x, y, z])
            return np.sum((spline_point - point) ** 2)

        # Perform the minimization over s in the interval [0, total_length]
        res = minimize_scalar(
            objective,
            bounds=(0, self.total_length),
            method="bounded",
            options={"xatol": 1e-8},
        )

        if res.success:
            s_closest = res.x % self.total_length
            x_closest = self.x_spline(s_closest)
            y_closest = self.y_spline(s_closest)
            z_closest = self.z_spline(s_closest)
            closest_point = np.array([x_closest, y_closest, z_closest])
            return closest_point, s_closest
        else:
            raise RuntimeError(
                "Optimization failed to find the closest point on the trajectory."
            )
