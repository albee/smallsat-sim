from argparse import Namespace
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils import xml_parser_rl


class VecEnv(BaseEnv):
    """
    Vectorized environment for the smallsat.
    """

    def __init__(self, args) -> None:
        # Flag to know whether VecEnv is being used
        self.using_rl = True

        # Number of environments running in parallel
        self.num_envs = self.env_cfg.control.RL.num_envs

        super().__init__(args)

        # Observation and action spaces
        self.obs_dim = 6
        self.act_dim = 8

        # Initial position and velocity
        self.init_qpos = self.mjx_batch.qpos
        self.init_qvel = self.mjx_batch.qvel

        # Perform a Just In Time compilation of mjx.step() so that it runs efficiently on GPU
        self.jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        self.jit_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
        self.reset()

    def reset(self) -> None:
        """
        Reset the agent in all the environment instances, while randomizing the initial position.
        """
        self.prev_shaping = None

        # Updating only qpos and qvel, not resetting all of mj_data
        self.mjx_data = self.mjx_data.replace(qpos=self.init_qpos[0])
        rng = jax.random.PRNGKey(np.random.randint(0, 9999))
        rng = jax.random.split(rng, self.num_envs)
        self.mjx_batch = jax.vmap(
            lambda rng: self.mjx_data.replace(
                qpos=jnp.concatenate(
                    [
                        self.mjx_data.qpos[0:2]
                        + jax.random.uniform(rng, (2,), minval=-1.0, maxval=1.0),
                        self.mjx_data.qpos[2].reshape(-1),
                        self._get_random_quaternion(rng)
                    ]
                )
            )
        )(rng)
        self.mjx_batch = self.mjx_batch.replace(qvel=self.init_qvel)
        self.mjx_batch = self.jit_forward(self.mjx_model, self.mjx_batch)

    def transition(
        self, actions: jnp.ndarray, states: jnp.ndarray
    ) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
        """
        Apply input action on the environment. Returns the rewards and wether the terminal state has been reached.
        """
        self.step(input=actions)

        # TODO: implement reward shaping for the case when the agent is out-of-bounds (collision corridor)

        # Penalize Euclidean distance from set point
        rewards = jnp.zeros(self.num_envs)
        shaping = -100 * jnp.sqrt(
            (states[:, 0]) ** 2 + (states[:, 1]) ** 2
        ) - 20 * jnp.sqrt(
            (states[:, 3]) ** 2
        )
        if self.prev_shaping is not None:
            rewards = shaping - self.prev_shaping
        self.prev_shaping = shaping

        # Check if the agent is out-of-bounds or has reached the goal
        is_terminal = jax.vmap(self._in_terminal_set)
        terminal = is_terminal(states)

        # Reward the agent for reaching the goal
        rewards += jnp.where(terminal, 1000, 0)

        return rewards, terminal

    def step(self, input) -> None:
        """
        Simulate environments for one timestep.
        """
        # Prepare env for simulation step
        self._pre_physics_step(input)

        # Advance simulation
        for substep in range(self.env_cfg.control.control_decimation):
            # Update viewer
            if (
                substep % self.env_cfg.viewer.viewer_decimation == 0
                and hasattr(self, "viewer")
                and self.viewer is not None
            ):
                mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)
                self._update_viewer()

            self.mjx_batch = self.jit_step(self.mjx_model, self.mjx_batch)

        # Execute post physics steps
        self._post_physics_step()

    def get_obs(self) -> jnp.ndarray:
        """
        Return all states.
        """
        # obs = [r (3),
        #        q (4),
        #        v (3), --> in BODY frame
        #        omega (3)]

        # Retrieve current rotation matrix
        R = self.mjx_batch.xmat[:, 1, :, :]

        # Rotate matrix
        R = jnp.transpose(R, (0, 2, 1))

        @jax.vmap
        def multiply_transpose_velocity(R, vel):
            return jnp.matmul(R, vel)  # Shape (3,)

        # Rotate intertial velocity to body velocity
        vel_body = multiply_transpose_velocity(R, self.mjx_batch.qvel[:, :3])

        # Create array of observations
        obs = jnp.concatenate(
            (self.mjx_batch.qpos, vel_body, self.mjx_batch.qvel[:, 3:]), axis=1
        )

        return obs

    def get_states(self, next_waypoint: jnp.ndarray) -> jnp.ndarray:
        """
        Return all states.
        """
        # Retrieve current rotation matrix
        R = self.mjx_batch.xmat[:, 1, :, :]

        # Rotate matrix
        R = jnp.transpose(R, (0, 2, 1))

        @jax.vmap
        def multiply_transpose_velocity(R, vel):
            return jnp.matmul(R, vel)  # Shape (3,)

        # Rotate intertial velocity to body velocity
        vel_body = multiply_transpose_velocity(R, self.mjx_batch.qvel[:, :3])

        # Compute rotation angle
        rot_angle = jnp.arctan2(R[:, 2, 1], R[:, 1, 1])

        # Compute the distance on each axis to the reference point
        delta_pos = self.mjx_batch.qpos[:, 0:3] - jnp.full(
            (self.num_envs, 3), next_waypoint
        )

        # Compute the attitude error
        ref_orientation = jnp.full(self.num_envs, jnp.pi / 2)
        delta_orientation = ref_orientation - rot_angle

        # Create array of observations
        states = jnp.concatenate(
            (delta_pos[:, 0:2], delta_orientation.reshape(-1, 1), vel_body[:, 0:2], self.mjx_batch.qvel[:, 5].reshape(-1, 1)), axis=1
        )

        return states

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

        # Batch the data and randomize the starting position
        rng = jax.random.PRNGKey(0)
        rng = jax.random.split(rng, self.num_envs)
        self.mjx_batch = jax.vmap(  # The initial position is randomized when the env is reset (at init and after each epoch)
            lambda rng: self.mjx_data.replace(qpos=self.mjx_data.qpos)
        )(
            rng
        )
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

        # Launch the renderer to create a video
        if args.video:
            self._create_renderer()
        else:
            # Same logic as for the viewer
            self.renderer = None
            self._update_renderer = lambda *args, **kwargs: None

    def _update_renderer(self):
        """
        Updates the renderer.
        """
        self.renderer.update_scene(self.data_vec[0], self.cam)
        sim_img = self.renderer.render().copy()
        self.frames.append(sim_img)

    def _pre_physics_step(self, input: jnp.ndarray) -> None:
        """
        Prepares the environment for the simulation step in MuJoCo.
        This includes:
            - Adding external disturbances
            - Adding perturbations to control input and model dynamics
            - ...
        """
        # External disturbances
        if self.disturbances:
            self.mjx_batch = self.mjx_batch.replace(
                qfrc_applied=self.disturbances.apply(self.data.time)
            )

        # Perturbations
        if self.perturbations:
            self.mjx_batch = self.mjx_batch.replace(
                ctrl=self.perturbations.apply(input, self.data.time)
            )
        else:
            self.mjx_batch = self.mjx_batch.replace(ctrl=input)

    def _in_terminal_set(self, delta_pos: jnp.ndarray) -> bool:
        """
        Returns one if in terminal set, zero otherwise.
        """
        # return jnp.all(jnp.abs(delta_pos) <= 0.5)
        return (
            jnp.sqrt(delta_pos[0] ** 2 + delta_pos[1] ** 2 + delta_pos[2] ** 2) <= 0.25
        )

    def _get_error_quaternion(self, q: jnp.ndarray, q_des: jnp.ndarray) -> jnp.ndarray:
        """
        Return the error between two quaternions.
        """
        # Quaternion error
        q_conj = jnp.array([q[0], -q[1], -q[2], -q[3]])
        e_q = jnp.array(
            [
                q_des[0] * q_conj[0]
                - q_des[1] * q_conj[1]
                - q_des[2] * q_conj[2]
                - q_des[3] * q_conj[3],
                q_des[0] * q_conj[1]
                + q_des[1] * q_conj[0]
                + q_des[2] * q_conj[3]
                - q_des[3] * q_conj[2],
                q_des[0] * q_conj[2]
                - q_des[1] * q_conj[3]
                + q_des[2] * q_conj[0]
                + q_des[3] * q_conj[1],
                q_des[0] * q_conj[3]
                + q_des[1] * q_conj[2]
                - q_des[2] * q_conj[1]
                + q_des[3] * q_conj[0],
            ]
        )

        # We only want to minimize eps part of error quaternion
        e_q = e_q[1:4]

        return e_q

    def _get_random_quaternion(self, rng) -> jnp.ndarray:
        """
        Return a random quaternion.
        """
        # Generate random samples from a uniform distribution over [0, 2π)
        theta = jax.random.uniform(rng, (1,)) * 2 * jnp.pi

        # Compute quaternion components
        w = jnp.cos(theta / 2)
        z = jnp.sin(theta / 2)

        quaternion = jnp.array([w, jnp.zeros_like(w), jnp.zeros_like(w), z]).reshape(-1)

        return quaternion
