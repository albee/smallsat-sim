import numpy as np
import yaml
import os
import mujoco
import mujoco.viewer

from smallsat_sim import SMALLSAT_SIM_ENVS_DIR
from smallsat_sim import SMALLSAT_SIM_LIB_DIR


class BaseEnv(object):
    def __init__(self, args) -> None:
        self._setup_sim(args)

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

    def _setup_sim(self, args):
        """
        This method sets up the correct simulation backend.
        Checks for args.use_mujoco
        """
        # Dynamically bind self.step() method
        if args.use_mujoco:
            self.step = self._step_mujoco
        else:
            self.step = self._step_casadi

        # Load the correct xml file
        smallsat = self.cfg['smallsat']['name']
        xml = os.path.join(SMALLSAT_SIM_LIB_DIR, smallsat, smallsat + ".xml")

        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        # Launch the viewer
        self._create_viewer()

    def _init_mujoco(self) -> None:
        """
        Sets up the simulation with the mujoco backend
        """
        # Load the correct xml file
        smallsat = self.cfg['smallsat']['name']
        xml = os.path.join(SMALLSAT_SIM_LIB_DIR, smallsat, smallsat + ".xml")

        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        # Launch the viewer
        self._create_viewer()

    def _init_casadi(self) -> None:
        """
        Sets up the simulation with the casadi backend
        """
        raise NotImplementedError("This function hasn't been implemented yet.")
    
    def _step_mujoco(self) -> None:
        """
        Uses mujoco physics engine to simulate system forward in time
        """
        # Step simulation
        # Controller callback is called internally to retrieve inputs
        # See: https://mujoco.readthedocs.io/en/latest/APIreference/APIglobals.html#mjcb-control
        mujoco.mj_step(self.model,self.data)

        # Update renderer
        self.viewer.sync()

    def _step_casadi(self) -> None:
        """
        Uses casadi to simulate system forward in time
        """
        raise NotImplementedError("This function hasn't been implemented yet.")
    