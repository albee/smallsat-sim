import numpy as np
import yaml
import os
import mujoco
import mujoco.viewer

from smallsat_sim import SMALLSAT_SIM_ENVS_DIR, SMALLSAT_SIM_LIB_DIR
from smallsat_sim.envs.dynamics import SymbolicModel

from argparse import Namespace


class BaseEnv(object):
    def __init__(self, args) -> None:
        # Setup simulation environment
        self._setup_sim(args)

        # Create symbolic model
        self.symbolic_model = SymbolicModel(self.lib_cfg)

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
        obs = self.get_obs()
        print("Success!")

        # Advance simulation
        mujoco.mj_step(self.model, self.data)

        # Update viewer
        self._update_viewer()

    def get_obs(self) -> np.array:
        """
        Return all states
        """
        # obs = [r (3),
        #        q (4),
        #        v (3),
        #        omega (3)]

        obs = np.concatenate((self.data.qpos, self.data.qvel))

        return obs

    def _create_viewer(self) -> None:
        """
        Creates a viewer to visualize simulation
        """
        self.viewer = mujoco.viewer.launch_passive(self.model, self.data)

    def _load_cfg(self, env_name: str, lib_name: str) -> dict:
        """
        Loads and returns the following config files:
            - env config file
            - lib config file
        """
        # Save env and lib names for later use
        self.env_name = env_name
        self.lib_name = lib_name

        # env config
        # localize relevant yaml file
        cfg_path = os.path.join(SMALLSAT_SIM_ENVS_DIR, env_name, "cfg", "config.yaml")

        # load from yaml file
        with open(cfg_path) as file:
            try:
                env_cfg = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        # lib config
        # localize relevant yaml file
        cfg_path = os.path.join(SMALLSAT_SIM_LIB_DIR, lib_name, "cfg", "config.yaml")

        # load from yaml file
        with open(cfg_path) as file:
            try:
                lib_cfg = yaml.safe_load(file)
            except yaml.YAMLError as exc:
                print(exc)

        return env_cfg, lib_cfg

    def _setup_sim(self, args: Namespace):
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Load the correct xml file
        xml = os.path.join(SMALLSAT_SIM_LIB_DIR, self.lib_name, self.lib_name + ".xml")

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
        """
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
