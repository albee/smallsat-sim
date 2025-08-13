from smallsat_sim.planners.mission.mission import MissionPlanner, Waypoint

import numpy as np

class MissionPlanner2DSquare(MissionPlanner):
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
            [0, -dist, astrobee_height],
            [dist/2, -dist, astrobee_height],
            [dist, -dist, astrobee_height],
            [dist, -dist/2, astrobee_height],
            [dist, 0, astrobee_height],
            [dist, dist/2, astrobee_height],
            [dist, dist, astrobee_height],
            [dist/2, dist, astrobee_height],
            [0, dist, astrobee_height],
            [-dist/2, dist, astrobee_height],
            [-dist, dist, astrobee_height],
            [-dist, dist/2, astrobee_height],
            [-dist, 0, astrobee_height],
            [-dist, -dist/2, astrobee_height],
            [-dist, -dist, astrobee_height],
            [-dist/2, -dist, astrobee_height]
        ]

        # Define attitude references (Euler angles)
        attitudes = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 45],
            [0, 0, 90],
            [0, 0, 90],
            [0, 0, 90],
            [0, 0, 135],
            [0, 0, 180],
            [0, 0, 180],
            [0, 0, 180],
            [0, 0, -135],
            [0, 0, -90],
            [0, 0, -90],
            [0, 0, -90],
            [0, 0, -45],
            [0, 0, 0]
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
            "Line",  # Point 16 to Point 1
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
