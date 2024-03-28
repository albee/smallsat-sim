import yaml
import os

from smallsat_sim import SMALLSAT_SIM_CONTROLLERS_DIR

class BaseController(object):
    def __init__(self) -> None:
        pass

    def controller_callback(self) -> None:
        """
        Defines the controller callback for the simulation step.
        """

    def _load_cfg(self, controller_name: str) -> dict:
        """
        Loads the config parameters from the file located in cfg/config.yaml    
        """
        # localize relevant yaml file
        cfg_path = os.path.join(SMALLSAT_SIM_CONTROLLERS_DIR, 
                                controller_name, "cfg", "config.yaml")

        # load from yaml file
        with open(cfg_path) as file:
            try:
                cfg = yaml.safe_load(file)
                return cfg
            except yaml.YAMLError as exc:
                print(exc)