import numpy as np
import mujoco
import mujoco.viewer

from smallsat_sim.envs.dynamics_2d import SymbolicModel2D

from smallsat_sim.envs.base_env import BaseEnv

from smallsat_sim.envs.dynamics_2d import SymbolicModel2D

from smallsat_sim.utils import xml_parser_2d

from argparse import Namespace


class BaseEnv2D(BaseEnv):

    def __init__(self, args):
        super().__init__(args)
        # Create symbolic model
        self.symbolic_model = SymbolicModel2D(self.model_cfg)
    
    def set_obs(self, v_frame: str = "body") -> np.ndarray:
        """
        Return all states

        args:
            v_frame (str): Specifies the frame of the velocity in the returned observation.
                             - "body": Return the velocity in the body frame.
                             - "inertial": Return the velocity in the inertial frame.

        returns:
            np.array: Array of observations containing position, orientation, velocity
                      and angular velocity.
        """
        # obs = [r (3),
        #        q (4),
        #        v (3), --> in body/inertial frame
        #        omega (3)]

        if v_frame == "body":

            # Retrieve current rotation matrix
            R = np.reshape(self.data.body("body0").xmat.copy(), (3, 3))

            # Rotate intertial velocity to body velocity
            v =  R.T @ self.data.cvel[-1, 3:6]

        elif v_frame == "inertial":

            # Retrieve inertial velocity from MuJoCo
            v = self.data.qvel[-1, :3].copy()

        else:
            raise RuntimeError(
                f"Specified velocity frame {v_frame} not valid. "
                "Must be either 'body' or 'inertial'."
            )

        # convert 3dof observation to 6dof
        omega = np.array([0, 0, self.data.qvel[-1]])
        obs = np.concatenate((self.data.xpos[-1], self.data.xquat[-1], v,  omega))

        # Save ground truth observations
        self.obs_gt = obs

        # Apply noise to observations
        self.obs = self._apply_obs_noise(obs)

    def _setup_sim(self, args: Namespace) -> None:
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Generate xml using env and model config files
        xml = xml_parser_2d.generate_mujoco_xml(self.env_cfg, self.model_cfg)
        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Launch the viewer
        if not args.headless:
            self._create_viewer()
        else:
            # If sim is run in headless mode, set the update_viewer method
            # to a lambda function which essentially does nothing
            self.viewer = None
            self._update_viewer = lambda *args, **kwargs: None

        # Launch the renderer to create a video
        if args.video:
            self._create_renderer()
        else:
            # Same logic as for the viewer
            self.renderer = None
            self._update_renderer = lambda *args, **kwargs: None

    def _create_viewer(self) -> None:
        """
        Creates a viewer to visualize simulation
        """
        # Create instance of MuJoCo viewer
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data, key_callback=self._key_callback
        )

        # Set default camera options
        self.viewer.cam.distance = 3.0
        # self.viewer.cam.trackbodyid = 2  # tracks smallsat
        self.viewer.cam.trackbodyid = 1  # tracks smallsat
        self.viewer.cam.azimuth = 10.0
        self.viewer.cam.type = 1