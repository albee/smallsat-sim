import numpy as np
import yaml
import os
import mujoco
import mujoco.viewer

from smallsat_sim import SMALLSAT_SIM_ENVS_DIR
from smallsat_sim import SMALLSAT_SIM_LIB_DIR

from argparse import Namespace

class BaseEnv(object):
    def __init__(self, args) -> None:
        self._setup_sim(args)

        # Initialize disturbance and perturbation to None as default setting
        self.disturbance = None
        self.perturbation = None

    def reset(self) -> None:
        """
        Resets environment to a desired state.
        """
        pass

    def step(self, input: np.array) -> None:
        """
        Simulate environment for one timestep.
        """
        # Prepare env for simulation step
        self._pre_physics_step(input)

        # Advance simulation
        mujoco.mj_step(self.model, self.data)        

        # Update viewer
        self._update_viewer()

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

    def _setup_sim(self, args: Namespace):
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Load the correct xml file
        smallsat = self.cfg['smallsat']['name']
        xml = os.path.join(SMALLSAT_SIM_LIB_DIR, smallsat, smallsat + ".xml")

        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)

        # Launch the viewer
        if not args.headless:
            self._create_viewer()
        else:
            # If sim is run in headless mode, set the update_viewer method
            # to a lambda function which essentially does nothing
            self._update_viewer = lambda *args, **kwargs: None
    
    def _update_viewer(self):
        """
        Updates the viewer
        """
        self.viewer.sync()

    def _pre_physics_step(self, input: np.ndarray) -> None:
        """"
        Prepares the environment for the simulation step in MuJoCo.
        This includes:
            - Adding external disturbances
            - Adding perturbations to control input and model dynamics
            - ...
        """
        # External disturbances
        if self.disturbance:
            self.data.qfrc_applied = self.disturbance.apply()

        # Perturbations
        if self.perturbation:
            self.data.ctrl = self.perturbation.apply()
        else:
            self.data.ctrl = input