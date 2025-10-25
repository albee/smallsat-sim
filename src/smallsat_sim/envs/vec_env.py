from argparse import Namespace
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx

# from mujoco.mjx import viewer
import wandb

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.utils import xml_parser_lightweight


class VecEnv(BaseEnv):
    """
    Vectorized environment for the smallsat.
    """

    def __init__(self, args) -> None:
        # Flag to know whether Weights & Biases should be used
        self.use_wandb = args.wandb

        # Number of environments running in parallel
        self.num_envs = self.env_cfg.control.RL.num_envs

        # Flag to decide whether to used the pretrained actor and critic networks
        self.use_pretrained = self.env_cfg.control.RL.use_pretrained

        # Flag to decide whether to use the adaptation module
        self.use_adaptive_approach = self.env_cfg.control.RL.use_adaptive_approach

        super().__init__(args)

        # Flag to know whether VecEnv is being used
        self.using_rl = True

        # Observation and action spaces
        self.obs_dim = 12
        self.act_dim = int(self.model.nu)
        if self.use_adaptive_approach is True:
            self.ext_dim = self.act_dim
        else:
            self.ext_dim = 0

        # Initial position and velocity
        self.init_qpos = self.mjx_batch.qpos
        self.init_qvel = self.mjx_batch.qvel

        # Max. offset from the initial position at the start
        self.max_start_offset = self.env_cfg.Bodies.max_start_offset

        # Perform a Just In Time compilation of mjx.step() so that it runs efficiently on GPU
        self.jit_step = jax.jit(jax.vmap(mjx.step, in_axes=(None, 0)))
        self.jit_forward = jax.jit(jax.vmap(mjx.forward, in_axes=(None, 0)))
        self.reset()

    def next_rng_keys(self, count: int = 1) -> jnp.ndarray:
        """
        Draw ``count`` fresh PRNG keys from the environment stream.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        splits = jax.random.split(self._rng, count + 1)
        self._rng = splits[0]
        return splits[1:]

    def reset(self) -> None:
        """
        Reset the agent in all the environment instances, while randomizing the initial position.
        """
        self.prev_shaping = None

        # Updating only qpos and qvel, not resetting all of mj_data
        self.mjx_data = self.mjx_data.replace(qpos=self.init_qpos[0])
        rng = self.next_rng_keys(self.num_envs)

        def _randomize_state(rng_key):
            pos_key, quat_key = jax.random.split(rng_key)
            random_pos = self.mjx_data.qpos[:3] + jax.random.uniform(
                pos_key,
                (3,),
                minval=-self.max_start_offset,
                maxval=self.max_start_offset,
            )
            random_quat = self._get_random_quaternion(quat_key)
            return self.mjx_data.replace(
                qpos=jnp.concatenate([random_pos, random_quat])
            )

        tmp_batch = jax.vmap(_randomize_state)(rng)
        self.mjx_batch = self.mjx_batch.replace(qpos=tmp_batch.qpos)
        self.mjx_batch = self.mjx_batch.replace(qvel=self.init_qvel)
        self.mjx_batch = self.jit_forward(self.mjx_model, self.mjx_batch)

    def transition(
        self,
        actions: jnp.ndarray,
        states: jnp.ndarray,
        iter: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply input action on the environment. Returns the rewards and wether the terminal state has been reached.
        """
        self.step(input=actions)

        # Reward shaping
        rewards = jnp.zeros(self.num_envs)

        pos_error = states[:, 0:3]
        att_error_vec = states[:, 3:6]
        body_vel = states[:, 6:9]
        ang_vel = states[:, 9:12]

        squared_pos_error = jnp.sum(pos_error**2, axis=1)
        manhattan_dist2goal = jnp.sum(jnp.abs(pos_error), axis=1)
        attitude_error_norm = jnp.linalg.norm(att_error_vec, axis=1)
        attitude_dev = jnp.exp(-1.0 * attitude_error_norm**2)
        squared_body_vel = jnp.sum(body_vel**2, axis=1)
        squared_angvel = jnp.sum(ang_vel**2, axis=1)
        control_effort = jnp.sum(actions**2, axis=1)

        # Curriculum-based training
        if (
            iter is not None and iter < 25
        ):  # Assumption: train for more than this many epochs
            shaping = (
                -3 * squared_pos_error
                - 2 * manhattan_dist2goal
                - 0.5 * squared_body_vel
                - 1 * squared_angvel
                + 0.1 * attitude_dev
                - 0.01 * control_effort
            )
        else:
            shaping = (
                -3 * squared_pos_error
                - 2 * manhattan_dist2goal
                - 0.5 * squared_body_vel
                - 1 * squared_angvel
                + 0.1 * attitude_dev
                - 0.01 * control_effort
            )

        # Penalize control actions outside of the bounds (approximate upper bound by largest possible value)
        lb_input = 0
        lb_mask = actions < lb_input
        shaping += 0.02 * jnp.sum((actions - lb_input) * lb_mask, axis=1)
        ub_input = 0.6
        ub_mask = actions > ub_input
        shaping -= 0.02 * jnp.sum((actions - ub_input) * ub_mask, axis=1)

        # Check if the agent is out-of-bounds or has reached the goal
        is_terminal = jax.vmap(self._in_terminal_set)
        terminal = is_terminal(states)

        # If the agent is in the terminal set, they should stop
        linear_speed_sq = jnp.sum(body_vel**2, axis=1)
        shaping += jnp.where(terminal, 0.1 * linear_speed_sq, 0)

        if self.prev_shaping is not None:
            rewards = shaping - self.prev_shaping
        self.prev_shaping = shaping

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
        Return state vector relative to the provided reference.
        """
        # Retrieve current rotation matrix
        R = self.mjx_batch.xmat[:, 1, :, :]

        # Rotate matrix
        R = jnp.transpose(R, (0, 2, 1))

        @jax.vmap
        def multiply_transpose_velocity(R, vel):
            return jnp.matmul(R, vel)  # Shape (3,)

        # Rotate inertial velocity to body velocity
        vel_body = multiply_transpose_velocity(R, self.mjx_batch.qvel[:, :3])

        # Ensure reference tensors have correct shape
        reference = jnp.asarray(next_waypoint)
        if reference.ndim == 1:
            reference = jnp.broadcast_to(reference, (self.num_envs, reference.shape[0]))

        # Extract position and quaternion reference
        ref_pos = reference[:, :3]
        ref_quat = reference[:, 3:7]

        delta_pos = self.mjx_batch.qpos[:, 0:3] - ref_pos

        get_error_quat = jax.vmap(self._get_error_quaternion, in_axes=(0, 0))
        attitude_error = get_error_quat(self.mjx_batch.qpos[:, 3:7], ref_quat)

        states = jnp.concatenate(
            (
                delta_pos,
                attitude_error,
                vel_body,
                self.mjx_batch.qvel[:, 3:6],
            ),
            axis=1,
        )

        return states

    def apply_random_perturbations(
        self,
        key,
        fraction_perturbed_envs: float,
        perturbation_distribution: jnp.ndarray = jnp.array(
            [0.5, 0.05, 0.15, 0.15, 0.15]
        ),
    ) -> None:
        """
        Apply a perturbation scenario to a subset of the environments (one per environment).
        NOTE: the thrusters are picked at random and the default times are 0.0 for now.
        """
        # Calculate number of environments to perturb
        num_perturbed = int(self.num_envs * fraction_perturbed_envs)
        if num_perturbed == 0:
            return

        # Split the key for permutation and for subkeys for perturbations
        perm_key, subkeys_key = jax.random.split(key, 2)
        subkeys = jax.random.split(subkeys_key, 5)

        # Get a random permutation of all environment indices
        env_indices = jnp.arange(self.num_envs)
        permuted_indices = jax.random.permutation(perm_key, env_indices)
        selected_indices = permuted_indices[:num_perturbed]

        # Determine number of environments for each perturbation based on distribution
        total_weight = jnp.sum(perturbation_distribution)  # should be 1.0
        base_counts = jnp.floor(
            num_perturbed * perturbation_distribution / total_weight
        ).astype(jnp.int32)
        count_sum = int(jnp.sum(base_counts))
        remainder = num_perturbed - count_sum

        # Compute fractional parts for extra allocation
        fractional_parts = (
            num_perturbed * perturbation_distribution / total_weight
        ) - base_counts
        sorted_indices = jnp.argsort(-fractional_parts)  # indices in descending order
        if remainder > 0:
            base_counts = base_counts.at[sorted_indices[:remainder]].add(1)

        # Partition the selected indices by counts into index arrays
        cumulative = 0
        indices_list = []
        for count in base_counts.tolist():
            indices_for_perturbation = selected_indices[cumulative : cumulative + count]
            cumulative += count
            indices_list.append(indices_for_perturbation)

        (
            stuck_off_thruster_envs,
            stuck_on_thruster_envs,
            faulty_valve_envs,
            saturated_thrust_envs,
            thrust_instability_envs,
        ) = indices_list

        # Apply the perturbations with respective subkeys
        self.perturbations.perturbations[0].stuck_off_thruster(
            subkeys[0], stuck_off_thruster_envs
        )
        self.perturbations.perturbations[1].stuck_on_thruster(
            subkeys[1], stuck_on_thruster_envs
        )
        self.perturbations.perturbations[2].register_perturbation(
            subkeys[2], faulty_valve_envs
        )
        self.perturbations.perturbations[3].register_perturbation(
            subkeys[3], saturated_thrust_envs
        )
        self.perturbations.perturbations[4].register_perturbation(
            subkeys[4], thrust_instability_envs
        )

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
        self.cam.trackbodyid = 1  # tracks smallsat
        self.cam.azimuth = 10.0
        self.cam.type = 1

        # Save frames to create the video
        self.frames = []

    def _setup_sim(self, args: Namespace):
        """
        Prepares simulation according to args.
        Creates a viewer depending on headless flag.
        """
        # Generate xml using env and model config files
        xml = xml_parser_lightweight.generate_mujoco_xml(self.env_cfg, self.model_cfg)

        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        self.mjx_model = mjx.put_model(
            self.model
        )  # impl='warp', warp requires a CUDA device & mujoco-mjx[warp]
        self.mjx_data = mjx.put_data(self.model, self.data)  # impl='warp'

        # Check that MJX puts the JAX arrays on GPU (it should do so automatically)
        print("Devices available to JAX: ", jax.devices())
        print("Device used by JAX: ", self.mjx_data.qpos.devices(), "\n")

        # Batch the data and randomize the starting position
        rng = self.next_rng_keys(self.num_envs)
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

    def _visualize_renderer(
        self, points: list[np.ndarray], color=[1, 0, 0, 2], size=[0.05, 0, 0]
    ) -> None:
        """
        Visualizes reference points in the MuJoCo renderer.
        """
        if (
            self.data.time >= self.env_cfg.renderer.start_recording
            and self.data.time <= self.env_cfg.renderer.end_recording
        ):
            self.renderer.scene.ngeom = 0

            # Update the renderer scene
            mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)
            self.renderer.update_scene(self.data_vec[0], self.cam)

            # Iterate over all points which need to be visualized in renderer
            for point in points:
                self.renderer.scene.ngeom += 1
                x = point[0]
                y = point[1]
                z = point[2]
                mujoco.mjv_initGeom(
                    self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=size,
                    pos=np.array([x, y, z]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(color),
                )

            # Extract image from renderer and append it for post-processing
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
                qfrc_applied=self.disturbances.apply(self.mjx_batch.time[0])
            )

        # Perturbations
        if self.perturbations:
            self.mjx_batch = self.mjx_batch.replace(
                ctrl=self.perturbations.apply(input, self.mjx_batch.time[0])
            )
        else:
            self.mjx_batch = self.mjx_batch.replace(ctrl=input)

    def _in_terminal_set(self, states: jnp.ndarray) -> jnp.ndarray:
        """
        Returns one if in terminal set, zero otherwise.
        """
        pos_error = states[0:3]
        eps = states[3:6]
        eps_norm = jnp.linalg.norm(eps)
        quat_scalar = jnp.sqrt(jnp.maximum(1.0 - jnp.minimum(1.0, eps_norm**2), 0.0))
        angle_error = 2 * jnp.arctan2(eps_norm, quat_scalar + 1e-8)
        pos_threshold = 0.2
        att_threshold = 0.05  # radians (approx 2.9 degrees)
        return jnp.logical_and(
            jnp.linalg.norm(pos_error) <= pos_threshold,
            angle_error <= att_threshold,
        )

    def _get_error_quaternion(
        self, q: jnp.ndarray, q_des: jnp.ndarray, eps=1e-12
    ) -> jnp.ndarray:
        """
        Compute the quaternion attitude error between the current and desired orientations.

        This function computes the right-invariant quaternion error q_e = q_des ⊗ conj(q),
        where both input quaternions are assumed to be in the [w, x, y, z] format.

        The result encodes the rotation that brings the current attitude `q` into alignment
        with the desired attitude `q_des`. The scalar part `w` represents cos(θ/2),
        and the vector part `e_signed` represents the rotation axis scaled by sin(θ/2),
        where θ is the shortest rotation angle between the two orientations.

        To ensure a unique and continuous representation (avoiding quaternion unwinding),
        the sign of the vector part is flipped whenever the scalar part is negative:
        e_signed = sign(w) * e.
        """
        q = q / jnp.maximum(jnp.linalg.norm(q), eps)
        q_des = q_des / jnp.maximum(jnp.linalg.norm(q_des), eps)
        qc = jnp.array([q[0], -q[1], -q[2], -q[3]])

        # Multiply q_des ⊗ qc
        w = q_des[0] * qc[0] - q_des[1] * qc[1] - q_des[2] * qc[2] - q_des[3] * qc[3]
        ex = q_des[0] * qc[1] + q_des[1] * qc[0] + q_des[2] * qc[3] - q_des[3] * qc[2]
        ey = q_des[0] * qc[2] - q_des[1] * qc[3] + q_des[2] * qc[0] + q_des[3] * qc[1]
        ez = q_des[0] * qc[3] + q_des[1] * qc[2] - q_des[2] * qc[1] + q_des[3] * qc[0]
        e = jnp.stack([ex, ey, ez])
        e_signed = jnp.where(w < 0.0, -e, e)

        return e_signed

    def _get_random_quaternion(self, rng) -> jnp.ndarray:
        """
        Return a random quaternion.
        """
        key, subkey1, subkey2 = jax.random.split(rng, 3)

        # Generate random samples from a uniform distribution over [0, 2π)
        theta1 = jax.random.uniform(key, (1,)) * 2 * jnp.pi
        theta2 = jax.random.uniform(subkey1, (1,)) * 2 * jnp.pi
        theta3 = jax.random.uniform(subkey2, (1,)) * 2 * jnp.pi

        # Compute quaternion components
        w = jnp.sin(theta1) * jnp.cos(theta2) * jnp.cos(theta3) + jnp.cos(
            theta1
        ) * jnp.sin(theta2) * jnp.sin(theta3)
        x = jnp.cos(theta1) * jnp.sin(theta2) * jnp.cos(theta3) - jnp.sin(
            theta1
        ) * jnp.cos(theta2) * jnp.sin(theta3)
        y = jnp.sin(theta1) * jnp.cos(theta2) * jnp.cos(theta3) - jnp.cos(
            theta1
        ) * jnp.sin(theta2) * jnp.sin(theta3)
        z = jnp.cos(theta1) * jnp.cos(theta2) * jnp.sin(theta3) + jnp.sin(
            theta1
        ) * jnp.sin(theta2) * jnp.cos(theta3)

        quaternion = jnp.array([w, x, y, z]).reshape(-1)
        quaternion = quaternion / jnp.linalg.norm(quaternion)

        return quaternion
