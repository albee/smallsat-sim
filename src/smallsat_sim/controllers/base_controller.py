from abc import abstractmethod, ABC
import numpy as np

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.planners.base_planner import BasePlanner


class BaseController(ABC):
    def __init__(self, env: BaseEnv, planner: BasePlanner, ctrl_cfg: object) -> None:

        # Initialize the planner module
        self.planner = planner

        # Set control decimation in respective environment configuration
        # NOTE: This works as ctrl_cfg is passed by reference
        env.env_cfg.control.control_decimation = ctrl_cfg.control_decimation

        # Copy logger to controller class if one is registered in env
        self.has_logger = False
        if hasattr(env, "logger"):
            self.logger = env.logger
            self.has_logger = True

    @abstractmethod
    def get_control_input(self, env: BaseEnv) -> None:
        """
        Returns the control input
        """
        pass

    @abstractmethod
    def _log(self, run_id: int, timestamp: float, obs: np.ndarray) -> None:
        """
        Logs desired quantities if flag is enabled
        """
        pass
