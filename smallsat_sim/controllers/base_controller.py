from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.planners.base_planner import BasePlanner

from abc import abstractmethod, ABC


class BaseController(ABC):
    def __init__(self, env: BaseEnv, planner: BasePlanner, ctrl_cfg: object) -> None:

        # Initialize the planner module
        self.planner = planner

        # Set control decimation in respective environment configuration
        # NOTE: This works as ctrl_cfg is passed by reference
        env.env_cfg.control.control_decimation = ctrl_cfg.control_decimation

    @abstractmethod
    def get_control_input(self, env: BaseEnv) -> None:
        """
        Returns the control input
        """
        pass
