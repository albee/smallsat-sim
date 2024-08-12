from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.planners.base_planner import BasePlanner
from smallsat_sim.controllers.base_controller import BaseController

from abc import abstractmethod


class BaseMPCController(BaseController):
    def __init__(self, env: BaseEnv, planner: BasePlanner, ctrl_cfg: object) -> None:
        # Flag to indicate whether the controller is MPC-based (for visualization purposes)
        planner.is_mpc = True

        super().__init__(env, planner, ctrl_cfg)

    @abstractmethod
    def get_control_input(self, env: BaseEnv) -> None:
        """
        Returns the control input
        """
        pass

    @abstractmethod
    def _visualize_prediction(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo viewer.
        """
        pass

    @abstractmethod
    def _visualize_prediction_renderer(self) -> None:
        """
        Plot predicted trajectory of MPC in MuJoCo renderer.
        """
        pass
