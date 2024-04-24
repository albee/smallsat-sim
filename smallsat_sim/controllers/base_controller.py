import yaml
import os

from smallsat_sim.envs.base_env import BaseEnv


class BaseController(object):
    def __init__(self) -> None:
        pass

    def get_control_input(self, env: BaseEnv) -> None:
        """
        Returns the control input
        """

    def _load_cfg(self, controller_name: str) -> object:
        """
        Loads the config parameters from the file located in cfg/config.yaml
        """
        # Dynamically import the correct model config module
        module = __import__(
            f"smallsat_sim.controllers.{controller_name}.cfg", fromlist=["config"]
        )
        ctrl_config = module.config.ControllerConfig()

        return ctrl_config
