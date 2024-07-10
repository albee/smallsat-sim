from smallsat_sim.planners.base_planner import BasePlanner

class MissionPlanner(BasePlanner):
    def __init__(self, env) -> None:
        super().__init__(env)

        # Load the waypoints
        self._load_waypoints()

    def _load_waypoints(self) -> None:
        """
        Loads the sparse waypoints that shall be reached
        """
        # Sets list of points
        self.waypoints = []

        # Visualize wapoints for DEBUG
        if True:
            self.visualize(self.waypoints)