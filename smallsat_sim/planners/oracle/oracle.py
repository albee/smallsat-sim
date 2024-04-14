from numpy import ndarray
from smallsat_sim.planners.base_planner import BasePlanner


class OraclePlanner(BasePlanner):
    def __init__(self, radius=20, spacing=0.25) -> None:
        super().__init__()
        self.radius = radius
        self.spacing = spacing

        # generate a circular reference trajectory around the gateway
        self._generate_reference()

    def get_reference(self, obs: ndarray) -> ndarray:
        """
        Dedicated method which is called externally
        """
        pass

    def _generate_reference(self):
        """
        Generates a circle in the z-dimension around the gateway
        """
        pass

    def _calc_world_point(self):
        pass
