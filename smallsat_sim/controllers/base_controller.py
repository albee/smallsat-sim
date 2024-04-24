import yaml
import os

from smallsat_sim.envs.base_env import BaseEnv


class BaseController(object):
    def __init__(self, env, planner, ctrl_cfg) -> None:

        # Initialize the planner module
        self.planner = planner

        # Set control decimation in respective environment configuration
        # NOTE: This works as ctrl_cfg is passed by reference
        env.env_cfg.control.control_decimation = ctrl_cfg.control_decimation

    def get_control_input(self, env: BaseEnv) -> None:
        """
        Returns the control input
        """
