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
