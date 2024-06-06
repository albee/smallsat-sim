import numpy as np
import mujoco
import mujoco.viewer
import cv2

from smallsat_sim.envs.dynamics import SymbolicModel
from smallsat_sim.utils import xml_parser
from smallsat_sim.utils.helpers import Rquat
from smallsat_sim.envs.perturbations import PerturbationList


from argparse import Namespace


class BaseEnv(object):
    def __init__(self, args) -> None:
        # Setup simulation environment
        self._setup_sim(args)

        # Create symbolic model
        self.symbolic_model = SymbolicModel(self.model_cfg)

        # Initialize disturbance and perturbation to None as default setting
        self.disturbance = None
        self.perturbations = None
        self.perturbations_keycodes = PerturbationList([]).keycode_dict.keys()

        # Initialize observations
        self.obs = self.get_obs()

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
        for substep in range(self.env_cfg.control.control_decimation):
            # Update viewer
            if substep % self.env_cfg.viewer.viewer_decimation == 0:
                self._update_viewer()
                self._update_renderer()

            # Step in MuJoCo engine
            mujoco.mj_step(self.model, self.data)

        # Execute post physics steps
        self._post_physics_step()

    def get_obs(self) -> np.array:
        """
        Return all states
        """
        # obs = [r (3),
        #        q (4),
        #        v (3), --> in BODY frame
        #        omega (3)]

        # Retrieve current rotation matrix
        R = np.reshape(self.data.body('body0').xmat, (3,3))

        # Rotate intertial velocity to body velocity
        vel_body = R.T @ self.data.qvel[:3]

        # Create array of observations
        obs = np.concatenate((self.data.qpos, vel_body, self.data.qvel[3:]))

        return obs
    
    def get_sim_rendering(self, output_filename: str) -> None:
        """
        Create a save a video rendering of the experiment.
        """
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        fps = 30
        height, width, _ = self.frames[0].shape
        
        video_writer = cv2.VideoWriter(output_filename + '.mp4', fourcc, fps, (width, height))

        for frame in self.frames:
            video_writer.write(frame)

        video_writer.release()

        print("Video saved as " + output_filename + ".mp4\n")

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
        self.viewer.cam.trackbodyid = 2  # tracks smallsat
        self.viewer.cam.azimuth = 10.0
        self.viewer.cam.type = 1

    def _create_renderer(self) -> None:
        """
        Creates a renderer to visualize the experiments (to later save them to a video).
        """
        # Create instance of MuJoCo renderer
        # self.renderer = mujoco.Renderer(self.model, width=1200, height=900)

        # Set up the scene and the default camera options
        self.scene = mujoco.MjvScene(self.model, 1000)
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 3.0
        self.cam.trackbodyid = 2  # tracks smallsat
        self.cam.azimuth = 10.0
        self.cam.type = 1

        # Create an offscreen framebuffer for rendering
        self.viewport = mujoco.MjrRect(0, 0, 1200, 900)

        # Save frames to create the video
        self.frames = []
        
    def _load_cfg(self, env_name: str, model_name: str) -> dict:
        """
        Loads and returns the following config files:
            - env config file
            - model config file
        """
        # Save env and model names for later use
        self.env_name = env_name
        self.model_name = model_name

        # Dynamically import the correct env config module
        module = __import__(f"smallsat_sim.envs.{env_name}.cfg", fromlist=["config"])
        env_cfg = module.config.EnvConfig()

        # Dynamically import the correct model config module
        module = __import__(f"smallsat_sim.model.{model_name}.cfg", fromlist=["config"])
        model_cfg = module.config.ModelConfig()

        return env_cfg, model_cfg

    def _setup_sim(self, args: Namespace):
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Generate xml using env and model config files
        xml = xml_parser.generate_mujoco_xml(self.env_cfg, self.model_cfg)
        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self._create_renderer()

        # Launch the viewer
        if not args.headless:
            self._create_viewer()
        else:
            # If sim is run in headless mode, set the update_viewer method
            # to a lambda function which essentially does nothing
            self.viewer = None
            self._update_viewer = lambda *args, **kwargs: None

    def _update_viewer(self):
        """
        Updates the viewer
        """
        self.viewer.sync()

    def _update_renderer(self):
        """
        Updates the renderer.
        """
        # self.renderer.update_scene(self.data)
        # sim_img = self.renderer.render().copy()
        # self.frames.append(sim_img)
        mujoco.mjv_updateScene(self.model, self.data, mujoco.MjvOption(), mujoco.MjvPerturb(), self.cam, mujoco.mjtCatBit.mjCAT_ALL, self.scene)
        # sim_img = self.renderer.render(self.scene).copy()
        # self.frames.append(sim_img)
        rgb_buffer = np.zeros((900, 1200, 3), dtype=np.uint8)
        upside_down_depth = np.empty((900, 1200, 1))
        mujoco.mjr_readPixels(rgb_buffer, upside_down_depth, self.viewport, mujoco.MjrContext(self.model, mujoco.mjtFontScale.mjFONTSCALE_100))
        rgb_buffer = np.flipud(rgb_buffer)
        self.frames.append(rgb_buffer)

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
        if self.perturbations:
            self.data.ctrl = self.perturbations.apply(input)
        else:
            self.data.ctrl = input

    def _post_physics_step(self) -> None:
        """
        Executes actions after stepping simulation
        """
        # Fetch most recent observations
        self.obs = self.get_obs()

    def _key_callback(self, keycode) -> None:
        """
        Callback function for keypressed detected in the MuJoCo viewer
        """
        try:
            if chr(keycode) in self.perturbations_keycodes:
                self.perturbations.key_callback(keycode)
        except:
            print("No perturbation list registered. Keycallback unsuccessful.")
