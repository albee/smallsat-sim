import numpy as np
import yaml
import os
import mujoco
import mujoco.viewer

from smallsat_sim import SMALLSAT_SIM_ENVS_DIR
from smallsat_sim import SMALLSAT_SIM_ROOT_DIR


class BaseEnv(object):
    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        """
        Resets environment to a desired state.
        """
        pass

    def step(self) -> None:
        """
        Simulate environment for one timestep.
        """
        pass

    def get_obs(self) -> np.array:
        """
        Return all states and optionally rewards
        """
        pass

    def _create_viewer(self) -> None:
        """
        Creates a viewer to visualize simulation
        """
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _load_cfg(self, env_name: str) -> dict:
        """
        Loads the config parameters from the file located in cfg/config.yaml    
        """
        # localize relevant yaml file
        cfg_path = os.path.join(SMALLSAT_SIM_ENVS_DIR, env_name, "cfg", "config.yaml")

        # load from yaml file
        with open(cfg_path) as file:
            try:
                cfg = yaml.safe_load(file)
                return cfg
            except yaml.YAMLError as exc:
                print(exc)


    # Rewards should be in here as well
    