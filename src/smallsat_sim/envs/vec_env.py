from argparse import Namespace
import numpy as np
import jax
import jax.numpy as jnp
import mujoco
import mujoco.viewer
from mujoco import mjx

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.disturbances import DisturbanceStatus
from smallsat_sim.utils import xml_parser_lightweight


class VecEnv(BaseEnv):
    """
    Vectorized environment for the smallsat.
    """

    def __init__(self, args) -> None:
        # Flag to know whether Weights & Biases should be used
        self.use_wandb = args.wandb

        # Run ID for logging
        self.run_id = self.env_cfg.control.RL.rl_run_id

        # Number of environments running in parallel
        self.num_envs = self.env_cfg.control.RL.num_envs

        # Flag to decide whether to enable random failures during training (and evaluation)
        self.train_with_failures = self.env_cfg.control.RL.train_with_failures

        # Flag to decide whether to used the pretrained actor and critic networks
        self.use_pretrained = self.env_cfg.control.RL.use_pretrained

        # Flag to decide whether to use the adaptation module
        self.use_adaptive_approach = self.env_cfg.control.RL.use_adaptive_approach

        # Adaptation module architecture
        self.am_architecture = self.env_cfg.control.RL.am_architecture
        self.history_len = self.env_cfg.control.RL.context_window_len

        super().__init__(args)

        # Mixer maps thruster commands to body-frame wrench
        mixer = jnp.asarray(self.symbolic_model.mixer, dtype=jnp.float32)
        self._thruster_mixer = jax.device_put(mixer)
        self._thruster_mixer_T = jax.device_put(mixer.T)

        # Flag to know whether VecEnv is being used
        self.using_rl = True

        # Observation and action spaces
        self.obs_dim = 12
        self.act_dim = int(self.model.nu)
        if self.use_adaptive_approach is True:
            self.ext_dim = 6
        else:
            self.ext_dim = 0
        self.res_dim = self.ext_dim

        # Initial position and velocity
        self.init_qpos = self.mjx_batch.qpos
        self.init_qvel = self.mjx_batch.qvel

        # Max. offset from the initial position at the start
        self.max_start_offset = self.env_cfg.Bodies.max_start_offset

        # Load mission tolerances, reward weights, and penalty weights
        self._load_vec_env_hyperparams()

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
            random_quat = self._get_random_quat(quat_key)
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
        next_waypoint: jnp.ndarray,
        iter: int | None = None,
    ) -> tuple[jnp.ndarray, jnp.ndarray]:
        """
        Apply input action on the environment. Returns the rewards and whether the terminal state has been reached.
        """

        def phi(st):
            """
            Potential for reward shaping.
            """
            # Reward kernels (normalize errors by tolerances)
            pos_term = jnp.exp(-jnp.sum((st[:, 0:3] / self.sigma_pos) ** 2, axis=1))
            vel_term = jnp.exp(-jnp.sum((st[:, 6:9] / self.sigma_vel) ** 2, axis=1))
            att_term = jnp.exp(
                -((jnp.linalg.norm(st[:, 3:6], axis=1) / self.sigma_att) ** 2)
            )
            ang_term = jnp.exp(-jnp.sum((st[:, 9:12] / self.sigma_angvel) ** 2, axis=1))

            return (
                self.w_pos * pos_term
                + self.w_vel * vel_term
                + self.w_att * att_term
                + self.w_angvel * ang_term
            )

        # Pre-step potential
        phi_s = phi(states)

        # Step the environment
        self.step(input=actions)

        # Get new states
        next_states = self.get_states(next_waypoint)

        # Post-step potential
        phi_s_next = phi(next_states)

        # Check if the agent is out-of-bounds or has reached the goal
        is_terminal = jax.vmap(self._in_terminal_set)
        terminal = is_terminal(next_states)

        # Penalties
        fuel_pen = jnp.sum(jnp.abs(actions), axis=1)
        lin_speed_sq = jnp.sum(next_states[:, 6:9] ** 2, axis=1)
        ang_speed_sq = jnp.sum(next_states[:, 9:12] ** 2, axis=1)
        vel_pen_terminal = jnp.where(
            terminal, self.lam_speed_terminal * lin_speed_sq, 0.0
        )
        angvel_pen_terminal = jnp.where(
            terminal, self.lam_ang_speed_terminal * ang_speed_sq, 0.0
        )
        fuel_pen_terminal = jnp.where(terminal, self.lam_fuel_terminal * fuel_pen, 0.0)
        penalties = (
            self.lam_fuel * fuel_pen
            + vel_pen_terminal
            + angvel_pen_terminal
            + fuel_pen_terminal
        )

        if self.use_adaptive_approach is True:
            # Wrench residual penalty
            residual_norm = jnp.linalg.norm(states[:, 12:18], axis=1)
            lam_residual = jnp.asarray(
                self.lam_wrench_residual, dtype=residual_norm.dtype
            )
            tolerance = jnp.asarray(
                self.wrench_residual_tolerance, dtype=residual_norm.dtype
            )
            clip_value = jnp.asarray(
                self.wrench_residual_clip, dtype=residual_norm.dtype
            )
            residual_excess = jnp.maximum(residual_norm - tolerance, 0.0)
            residual_clipped = jnp.minimum(residual_excess, clip_value)
            wrench_residual_pen = lam_residual * residual_clipped
            penalties = penalties + wrench_residual_pen

        # Reward shaping
        rewards = phi_s_next - phi_s - penalties

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

        get_error_quat = jax.vmap(self._get_error_quat_logvec, in_axes=(0, 0))
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

    def apply_random_disturbance(self, key, fraction_disturbed_envs: float) -> None:
        """
        Apply the constant force disturbance to a subset of environments.
        """
        if self.disturbances is None:
            return

        clamped_fraction = max(0.0, min(1.0, fraction_disturbed_envs))
        num_disturbed = int(self.num_envs * clamped_fraction)
        if num_disturbed == 0:
            return

        env_indices = jnp.arange(self.num_envs)
        permuted_indices = jax.random.permutation(key, env_indices)
        selected_indices = permuted_indices[:num_disturbed]

        constant_force_disturbance = None
        for disturbance in self.disturbances.disturbances:
            if (
                getattr(disturbance, "failure_type", None)
                == DisturbanceStatus.CONSTANT_FORCE
            ):
                constant_force_disturbance = disturbance
                break

        if constant_force_disturbance is None:
            return

        constant_force_disturbance.const_force_disturbance(selected_indices)

    def get_desired_wrench(self, ctrl: jnp.ndarray) -> jnp.ndarray:
        """
        Compute the net body-frame wrench generated by the commanded thruster forces.
        """
        ctrl = jnp.asarray(ctrl, dtype=self._thruster_mixer_T.dtype)
        ctrl = jnp.atleast_2d(ctrl)
        wrench = ctrl @ self._thruster_mixer_T
        return wrench

    def get_actual_wrench(self) -> jnp.ndarray:
        """
        Return the body-frame wrench computed from the forces actually applied by MuJoCo.
        """
        actuator_force = jnp.asarray(
            self.mjx_batch.actuator_force, dtype=self._thruster_mixer_T.dtype
        )
        actuator_force = jnp.atleast_2d(actuator_force)
        wrench = actuator_force @ self._thruster_mixer_T
        return wrench

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
        return jnp.linalg.norm(states[0:3]) <= self.sigma_pos

    def _get_error_quat_logvec(
        self, q: jnp.ndarray, q_des: jnp.ndarray, eps: float = 1e-9
    ) -> jnp.ndarray:
        """
        Return SO(3) log-map (rotation vector) that rotates q -> q_des.
        """

        # Normalize
        def _unit(a):
            return a / (jnp.linalg.norm(a) + eps)

        q = _unit(q)
        q_des = _unit(q_des)

        # Error quaternion: q_e = q_des ⊗ conj(q)
        w, x, y, z = q
        qc = jnp.array([w, -x, -y, -z])
        w2, x2, y2, z2 = q_des
        we = w2 * qc[0] - x2 * qc[1] - y2 * qc[2] - z2 * qc[3]
        ex = w2 * qc[1] + x2 * qc[0] + y2 * qc[3] - z2 * qc[2]
        ey = w2 * qc[2] - x2 * qc[3] + y2 * qc[0] + z2 * qc[1]
        ez = w2 * qc[3] + x2 * qc[2] - y2 * qc[1] + z2 * qc[0]
        e = jnp.stack([ex, ey, ez])

        # Enforce shortest path / continuity: flip WHOLE quaternion if we < 0
        sign = jnp.where(we < 0.0, -1.0, 1.0)
        we = sign * we
        e = sign * e

        # log-map: axis * theta, with ||logvec|| = theta in [0, pi]
        e_norm = jnp.linalg.norm(e)  # == sin(theta/2)
        we_abs = jnp.clip(jnp.abs(we), 0.0, 1.0)
        theta = 2.0 * jnp.arctan2(e_norm, we_abs)
        axis = e / (e_norm + eps)
        logvec = axis * theta

        return logvec

    def _get_random_quat(self, rng) -> jnp.ndarray:
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

    def _load_vec_env_hyperparams(self) -> None:
        """
        Load VecEnv-specific hyperparams.
        """
        # Mission tolerances
        self.sigma_pos = self.env_cfg.control.RL.sigma_pos
        self.sigma_vel = self.env_cfg.control.RL.sigma_vel
        self.sigma_att = self.env_cfg.control.RL.sigma_att
        self.sigma_angvel = self.env_cfg.control.RL.sigma_angvel

        # Reward weights
        self.w_pos = self.env_cfg.control.RL.w_pos
        self.w_vel = self.env_cfg.control.RL.w_vel
        self.w_att = self.env_cfg.control.RL.w_att
        self.w_angvel = self.env_cfg.control.RL.w_angvel

        # Penalty weights
        self.lam_fuel = self.env_cfg.control.RL.lam_fuel
        self.lam_speed_terminal = self.env_cfg.control.RL.lam_speed_terminal
        self.lam_ang_speed_terminal = self.env_cfg.control.RL.lam_ang_speed_terminal
        self.lam_fuel_terminal = self.env_cfg.control.RL.lam_fuel_terminal
        self.lam_wrench_residual = self.env_cfg.control.RL.lam_wrench_residual
        self.wrench_residual_tolerance = (
            self.env_cfg.control.RL.wrench_residual_tolerance
        )
        self.wrench_residual_clip = self.env_cfg.control.RL.wrench_residual_clip
