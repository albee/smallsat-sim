from argparse import Namespace
import numpy as np
import mujoco
import mujoco.viewer
from mujoco import mjx
import jax
import jax.numpy as jnp

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils import xml_parser_rl


class VecEnv(BaseEnv):
    """
    Vectorized environment for the smallsat.
    """
    def __init__(self, args) -> None:
        super().__init__(args)

        # Number of environments running in parallel
        self.n_envs = 4096

        # Initial position and velocity
        self.init_qpos = self.data.qpos # TODO: update to use MJX instead?
        self.init_qvel = self.data.qvel

        # Perform a Just In Time compilation of mjx.step() so that it runs efficiently on GPU
        self.jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))

        self.reset()

    def reset(self) -> None:
        """
        Reset the agent to the initial state in all the environment instances.
        """
        self.prev_shaping = None # TODO: adapt this to batched environments

        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.init_qpos
        self.data.qvel[:] = self.init_qvel
        mujoco.forward(self.model, self.data)

    # def transition(self, actions: torch.tensor) -> tuple[torch.tensor, torch.tensor, torch.tensor]:
    #     """
    #     Apply input action on the environment. Returns the states, rewards and wether the terminal state has been reached.
    #     """
    #     self.step(input=ctrl_input) # TODO: think about what the main loop lokks like and decide how to handle this

    #     rewards = torch.zeros(1)
    #     shaping = torch.zeros(1) # TODO: implement reward shaping
    #     if self.prev_shaping is not None:
    #         rewards = shaping - self.prev_shaping
    #     self.prev_shaping = shaping

    #     terminal = torch.zeros(1, dtype=bool) # TODO: implement "game over" checking

    #     return states, rewards, terminal

    def step(self, input) -> None:
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

            self.batch = self.jit_step(self.mjx_model, self.batch)

        # Print some information for debugging
        print(f"Time: {self.batch.time[0]} and Pos = {self.batch.qpos[0]}")

        # Execute post physics steps
        self._post_physics_step()

    def get_obs(self) -> np.array:
        """
        Return all states.
        """
        # obs = [r (3),
        #        q (4),
        #        v (3), --> in BODY frame
        #        omega (3)]

        # Retrieve current rotation matrix
        R = self.batch.xmat[:,1,:,:]

        # Rotate matrix
        R = jnp.transpose(R, (0,2,1))

        @jax.vmap
        def multiply_transpose_velocity(R, vel):
            return jnp.matmul(R, vel)  # Shape (3,)

        # Rotate intertial velocity to body velocity
        vel_body = multiply_transpose_velocity(R, self.batch.qvel[:,:3])

        # Create array of observations
        obs = jnp.concatenate((self.batch.qpos, vel_body, self.batch.qvel[:,3:]), axis=1)

        return obs
    
    def _create_viewer(self, args) -> None:
        """
        Creates a viewer to visualize simulation
        """
        # self.viewer = None

        # Create instance of MuJoCo viewer
        self.viewer = mujoco.viewer.launch_passive(
            self.model, self.batched_mj_data[0], key_callback=self._key_callback
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

        # Batch the data
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, self.n_envs)
        self.batch = jax.vmap(lambda rng: self.mjx_data.replace(qpos=self.mjx_data.qpos))(rng)
        self.batched_mj_data = mjx.get_data(self.model, self.batch)
        mjx.get_data_into(self.batched_mj_data, self.model, self.batch)

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
                self.mjx_data = self.mjx_data.replace(ctrl=jax.numpy.asarray(self.perturbations.apply(input)))
                self.data = mjx.get_data(self.model, self.mjx_data)
        else:
                self.batch = self.batch.replace(ctrl=jax.numpy.asarray(input))
                # self.batched_mj_data = mjx.get_data(self.model, self.batch)
                mjx.get_data_into(self.batched_mj_data, self.model, self.batch)

        print(f"Input: {input}")

    def _key_callback(self, keycode) -> None:
        """
        Callback function for keypressed detected in the MuJoCo viewer
        """
        raise NotImplementedError("Key callbacks are not supported for simulation in parallel.")
