from smallsat_sim.planners.mission.mission import MissionPlanner, Waypoint

import numpy as np

class MissionPlanner2DxLine(MissionPlanner):
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
    def _load_waypoints(self) -> None:
        """
        Loads the sparse waypoints that shall be reached
        """
        # Define positional references
        dist = 0.7
        astrobee_height = 0.4
        positions = [
            [-0.8, 0.0, astrobee_height],
            [-0.6, 0.0, astrobee_height],
            [-0.4, 0.0, astrobee_height],
            [-0.2, 0.0, astrobee_height]
         ]
        # Define attitude references (Euler angles)
        attitudes = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
            ]
        # Define connection type between waypoints
        segment_types = [
            "Line",  # Point 1 to Point 2
            "Line",  # Point 2 to Point 3
            "Line",  # Point 3 to Point 4
            "Line",  # Point 4 to Point 1
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
