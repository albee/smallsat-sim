from argparse import Namespace
import numpy as np
import torch
import mujoco
import mujoco.viewer
from mujoco import mjx
import jax
import jax.numpy as jnp

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils import xml_parser_rl
from smallsat_sim.utils.helpers import jax_to_torch


class VecEnv(BaseEnv):
    """
    Vectorized environment for the smallsat.
    """
    def __init__(self, args) -> None:
        # Use GPU acceleration if available
        print("GPU available: ", torch.cuda.is_available()) 
        print("Number of GPUs: ", torch.cuda.device_count())
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Number of environments running in parallel
        self.n_envs = 1

        super().__init__(args)

        # Observation and action spaces
        self.obs_dim = 13
        self.act_dim = 12

        # Initial position and velocity
        self.init_qpos = self.mjx_batch.qpos
        self.init_qvel = self.mjx_batch.qvel

        # Perform a Just In Time compilation of mjx.step() so that it runs efficiently on GPU
        self.jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        self.jit_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
        self.reset()

    def reset(self) -> None:
        """
        Reset the agent to the initial state in all the environment instances.
        """
        self.prev_shaping = None

        # TODO: do we need to reset all of the MuJoCo data or is updating qpos and qvel enough?
        self.mjx_batch.replace(qpos=self.init_qpos)
        self.mjx_batch.replace(qvel=self.init_qvel)
        self.mjx_batch = self.jit_forward(self.mjx_model, self.mjx_batch)

    def transition(self, actions: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
        """
        Apply input action on the environment. Returns the states, rewards and wether the terminal state has been reached.
        """
        self.step(input=actions.numpy())

        # TODO: implement reward shaping
        rewards = torch.zeros(self.n_envs, device=self.device)
        shaping = torch.zeros(self.n_envs, device=self.device)
        if self.prev_shaping is not None:
            rewards = shaping - self.prev_shaping
        self.prev_shaping = shaping

        terminal = torch.zeros(self.n_envs, dtype=bool, device=self.device) # TODO: implement "game over" checking

        return self.obs, rewards, terminal

    def step(self, input) -> None:
        """
        Simulate environments for one timestep.
        """
        # Prepare env for simulation step
        self._pre_physics_step(input)

        # Advance simulation
        for substep in range(self.env_cfg.control.control_decimation):
            # Update viewer
            if substep % self.env_cfg.viewer.viewer_decimation == 0:
                mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)
                self._update_viewer()

            self.mjx_batch = self.jit_step(self.mjx_model, self.mjx_batch)

        # Print some information for debugging
        # print(f"Time: {self.mjx_batch.time[0]} and Pos = {self.mjx_batch.qpos[0]}")

        # Execute post physics steps
        self._post_physics_step()

    def get_obs(self) -> torch.tensor:
        """
        Return all states.
        """
        # obs = [r (3),
        #        q (4),
        #        v (3), --> in BODY frame
        #        omega (3)]

        # Retrieve current rotation matrix
        R = self.mjx_batch.xmat[:,1,:,:]

        # Rotate matrix
        R = jnp.transpose(R, (0,2,1))

        @jax.vmap
        def multiply_transpose_velocity(R, vel):
            return jnp.matmul(R, vel)  # Shape (3,)

        # Rotate intertial velocity to body velocity
        vel_body = multiply_transpose_velocity(R, self.mjx_batch.qvel[:,:3])

        # Create array of observations
        obs_jax_ = jnp.concatenate((self.mjx_batch.qpos, vel_body, self.mjx_batch.qvel[:,3:]), axis=1)

        # Convert obs to PyTorch
        obs = jax_to_torch(obs_jax_, self.device)

        return obs
    
    def _create_viewer(self, args) -> None:
        """
        Creates a viewer to visualize simulation
        """
        # Create instance of MuJoCo viewer
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.data_vec[0], key_callback=self._key_callback
        )

        # Set default camera options
        self.viewer.cam.distance = 3.0
        self.viewer.cam.trackbodyid = 1  # tracks smallsat
        self.viewer.cam.azimuth = 10.0
        self.viewer.cam.type = 1

    def _setup_sim(self, args: Namespace):
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Generate xml using env and model config files
        xml = xml_parser_rl.generate_mujoco_xml(self.env_cfg, self.model_cfg)

        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.mjx_model = mjx.put_model(self.model)
        self.mjx_data = mjx.put_data(self.model, self.data)

        # Check that MJX puts the JAX arrays on GPU (it should do so automatically)
        print("Devices available to JAX: ", jax.devices())
        print("Device used by JAX: ", self.mjx_data.qpos.devices(), "\n")

        # Batch the data
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, self.n_envs)
        self.mjx_batch = jax.vmap(lambda rng: self.mjx_data.replace(qpos=self.mjx_data.qpos))(rng)
        self.data_vec = mjx.get_data(self.model, self.mjx_batch)
        mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)

        # Launch the viewer
        if not args.headless:
            self._create_viewer(args)
        else:
            # If sim is run in headless mode, set the update_viewer method
            # to a lambda function which essentially does nothing
            self.viewer = None
            self._update_viewer = lambda *args, **kwargs: None

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
                self.mjx_batch = self.mjx_batch.replace(ctrl=jax.numpy.asarray(self.perturbations.apply(input)))
        else:
                self.mjx_batch = self.mjx_batch.replace(ctrl=jax.numpy.asarray(input))

        # Print the control inputs for debugging
        # print(f"Input: {input}")

    def _key_callback(self, keycode) -> None:
        """
        Callback function for keypressed detected in the MuJoCo viewer
        """
        raise NotImplementedError("Key callbacks are not supported for simulation in parallel.")
