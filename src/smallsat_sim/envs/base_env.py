import numpy as np
import mujoco
import mujoco.viewer
import cv2
import os
import jax
import jax.numpy as jnp
from datetime import datetime

from smallsat_sim.envs.dynamics import SymbolicModel
from smallsat_sim.utils import xml_parser
from smallsat_sim.envs.disturbances import DisturbanceList
from smallsat_sim.envs.perturbations import PerturbationList
from smallsat_sim.utils.logger import Logger
from smallsat_sim import SMALLSAT_STEWARD_ROOT_DIR


from argparse import Namespace


class BaseEnv(object):
    def __init__(self, args: Namespace) -> None:
        # Setup simulation environment
        self._setup_sim(args)

        # Create symbolic model
        self.symbolic_model = SymbolicModel(self.model_cfg)

        # Initialize disturbance and perturbation to None as default setting
        self.disturbances = None
        self.disturbances_keycodes = DisturbanceList([]).keycode_dict.keys()
        self.perturbations = None
        self.perturbations_keycodes = PerturbationList([]).keycode_dict.keys()

        # Initialize observations
        self.set_obs(v_frame=self.env_cfg.sim.obs.v_frame)

        # Initialize arguments
        self.args = args

        # Save time of simulation start (for filenames)
        self.sim_start_time = datetime.now().strftime("%Y-%m-%d_%H:%M:%S")

        # Initialize MC iterations
        self.run_id = 0

        # Initialize logger if enabled
        if args.log:
            self.logger = Logger(log_name=self.sim_start_time)

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
                # self._update_renderer() # Moved to planner to visualize ref. as well

            # Step in MuJoCo engine
            mujoco.mj_step(self.model, self.data)

        # Execute post physics steps
        self._post_physics_step()

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

        obs = np.concatenate((self.data.xpos[-1], self.data.xquat[-1], v,  self.data.qvel[3:6]))

        # Save ground truth observations
        self.obs_gt = obs

        # Apply noise to observations
        self.obs = self._apply_obs_noise(obs)

    def get_obs(self) -> np.ndarray | jnp.ndarray:
        """
        Returns the current (noisy) observations
        """
        return self.obs.copy()

    def get_obs_gt(self) -> np.ndarray | jnp.ndarray:
        """
        Return the GT observations. Use this method for
        visualization and for evaluations.
        """
        return self.obs_gt.copy()

    def _apply_obs_noise(
        self, obs: np.ndarray | jnp.ndarray
    ) -> np.ndarray | jnp.ndarray:
        """
        Applies additive Gaussian noise on top of observations.
        For MuJoCo: obs are of type np.ndarray
        For MJX: obs are of type
        """
        if self.env_cfg.sim.noise.add_obs_noise:
            # Single agent in MuJoCo
            if isinstance(obs, np.ndarray):
                # Calculate noise
                noise_r = np.random.normal(0, self.env_cfg.sim.noise.sigma_r, 3)
                noise_q = np.random.normal(0, self.env_cfg.sim.noise.sigma_q, 4)
                noise_v = np.random.normal(0, self.env_cfg.sim.noise.sigma_v, 3)
                noise_w = np.random.normal(0, self.env_cfg.sim.noise.sigma_w, 3)

                # Concatenate noise
                noise = np.concatenate((noise_r, noise_q, noise_v, noise_w))

                # Add noise to observations and return
                return obs + noise

            # Multiple agents in MJX
            else:
                # Fix a random seed for the PRNG key and set the key
                random_seed = np.random.randint(0, high=9999, size=4)

                # Calculate noise for each environment
                num_envs = obs.shape[0]
                noise_r = jax.random.multivariate_normal(
                    jax.random.PRNGKey(random_seed[0]),
                    jnp.zeros((num_envs, 3)),
                    self.env_cfg.sim.noise.sigma_r * jnp.identity(3),
                )
                noise_q = jax.random.multivariate_normal(
                    jax.random.PRNGKey(random_seed[0]),
                    jnp.zeros((num_envs, 4)),
                    self.env_cfg.sim.noise.sigma_q * jnp.identity(4),
                )
                noise_v = jax.random.multivariate_normal(
                    jax.random.PRNGKey(random_seed[0]),
                    jnp.zeros((num_envs, 3)),
                    self.env_cfg.sim.noise.sigma_v * jnp.identity(3),
                )
                noise_w = jax.random.multivariate_normal(
                    jax.random.PRNGKey(random_seed[0]),
                    jnp.zeros((num_envs, 3)),
                    self.env_cfg.sim.noise.sigma_w * jnp.identity(3),
                )

                # Concatenate noise along the second dimension
                noise = jnp.concatenate((noise_r, noise_q, noise_v, noise_w), axis=1)

                # Add noise to observations and return
                return obs + noise
        else:
            return obs

    def get_sim_rendering(self, output_filename: str) -> None:
        """
        Create and save a video rendering of the experiment.
        """
        if not self.args.video:
            return

        # Define video settings
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        fps = self.env_cfg.renderer.fps
        height, width, _ = self.frames[0].shape

        # Define save settings
        video_dir = os.path.join(SMALLSAT_STEWARD_ROOT_DIR, "videos")
        os.makedirs(video_dir, exist_ok=True)  # Ensure the video directory exists

        video_path = os.path.join(
            video_dir,
            output_filename + "_" + self.sim_start_time + ".mp4",
        )

        video_writer = cv2.VideoWriter(
            video_path,
            fourcc,
            fps,
            (width, height),
        )

        # Parse video frame by frame
        for frame in self.frames:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            video_writer.write(frame)

        # Save video at specified location
        video_writer.release()

        print(f"Video saved as {video_path}\n")

        # Reset frames to free up memory
        self.frames = []

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
        self.renderer = mujoco.Renderer(
            self.model,
            width=self.env_cfg.renderer.width,
            height=self.env_cfg.renderer.height,
        )

        # Set up the scene and the default camera options
        self.cam = mujoco.MjvCamera()
        self.cam.distance = 5.0
        self.cam.trackbodyid = 2  # tracks smallsat
        self.cam.azimuth = 10.0
        self.cam.type = 1

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

    def _setup_sim(self, args: Namespace) -> None:
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Generate xml using env and model config files
        xml = xml_parser.generate_mujoco_xml(self.env_cfg, self.model_cfg)
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

    def _update_viewer(self) -> None:
        """
        Updates the viewer
        """
        self.viewer.sync()

    def _update_renderer(self) -> None:
        """
        Updates the renderer.
        """
        self.renderer.update_scene(self.data, self.cam)
        sim_img = self.renderer.render().copy()
        self.frames.append(sim_img)

    def _pre_physics_step(self, input: np.ndarray) -> None:
        """
        Prepares the environment for the simulation step in MuJoCo.
        This includes:
            - Adding external disturbances
            - Adding perturbations to control input and model dynamics
            - ...
        """
        # External disturbances
        if self.disturbances:
            self.data.qfrc_applied = self.disturbances.apply()

        # Perturbations
        if self.perturbations:
            self.data.ctrl = self.perturbations.apply(input, self.data.time)
        else:
            self.data.ctrl = input

        # Log perturbed inputs
        if hasattr(self, "logger"):
            self.logger.log(self.run_id, self.data.time, u_actual=self.data.ctrl.copy())

    def _post_physics_step(self) -> None:
        """
        Executes actions after stepping simulation
        """
        # Fetch most recent observations
        self.set_obs(v_frame=self.env_cfg.sim.obs.v_frame)

    def _key_callback(self, keycode) -> None:
        """
        Callback function for keypressed detected in the MuJoCo viewer
        """
        try:
            if chr(keycode) in self.perturbations_keycodes:
                self.perturbations.key_callback(keycode)
            if chr(keycode) in self.disturbances_keycodes:
                self.disturbances.key_callback(keycode)
        except:
            print(
                "No disturbance or perturbation list registered. Keycallback unsuccessful."
            )
