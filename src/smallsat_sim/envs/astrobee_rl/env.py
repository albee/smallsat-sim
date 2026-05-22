import jax.numpy as jnp

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.envs.perturbations_rl import (
    Perturbation,
    PerturbationList,
    StuckOffThrusters,
    StuckOnThrusters,
    FaultyValve,
    SaturatedThrust,
    ThrustInstability,
)
from smallsat_sim.envs.disturbances import DisturbanceList, ConstantForceDisturbance


class AstrobeeEnvVectorized(VecEnv):
    def __init__(
        self,
        args,
        run_name: str = "default",
        init_pos: jnp.ndarray | None = None,
        max_start_offset: float | None = None,
        train_with_failures: bool | None = None,
        use_pretrained: bool | None = None,
        use_adaptive_approach: bool | None = None,
        am_architecture: str | None = None,
        adaptive_context_mode: str | None = None,
        use_task_conditioned_am: bool | None = None,
        am_predict_delta_weight: float | None = None,
        am_predict_tracking_weight: float | None = None,
    ) -> None:
        # Run name for logging
        self.run_name = run_name

        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee_rl", model_name="astrobee"
        )
        if init_pos is not None:
            self.env_cfg.Bodies.bodies_list[0].pos = init_pos
        if max_start_offset is not None:
            self.env_cfg.Bodies.max_start_offset = max_start_offset
        if train_with_failures is not None:
            self.env_cfg.control.RL.train_with_failures = train_with_failures
        if use_pretrained is not None:
            self.env_cfg.control.RL.use_pretrained = use_pretrained
        if use_adaptive_approach is not None:
            self.env_cfg.control.RL.use_adaptive_approach = use_adaptive_approach
        if am_architecture is not None:
            self.env_cfg.control.RL.am_architecture = am_architecture
        if adaptive_context_mode is not None:
            self.env_cfg.control.RL.adaptive_context_mode = adaptive_context_mode
        if use_task_conditioned_am is not None:
            self.env_cfg.control.RL.use_task_conditioned_am = use_task_conditioned_am
        if am_predict_delta_weight is not None:
            self.env_cfg.control.RL.am_predict_delta_weight = am_predict_delta_weight
        if am_predict_tracking_weight is not None:
            self.env_cfg.control.RL.am_predict_tracking_weight = (
                am_predict_tracking_weight
            )
        super().__init__(args=args)

        # Instantiate perturbations
        # Ensure a clean shared mask when constructing a new vectorized env.
        Perturbation.thruster_mask = None
        perturbation_keys = self.next_rng_keys(5)
        self.perturbations = PerturbationList(
            [
                StuckOffThrusters(
                    self.env_cfg, self.model_cfg, perturbation_keys[0]
                ),  # 0
                StuckOnThrusters(
                    self.env_cfg, self.model_cfg, perturbation_keys[1]
                ),  # 1
                FaultyValve(self.env_cfg, self.model_cfg, perturbation_keys[2]),  # 2
                SaturatedThrust(
                    self.env_cfg, self.model_cfg, perturbation_keys[3]
                ),  # 3
                ThrustInstability(
                    self.env_cfg, self.model_cfg, perturbation_keys[4]
                ),  # 4
            ]
        )

        # Instantiate disturbances
        disturbance_key = self.next_rng_keys(1)[0]
        self.disturbances = DisturbanceList(
            [ConstantForceDisturbance(self.env_cfg, disturbance_key)]
        )
        self._refresh_effect_states()

    def reset_perturbations(self) -> None:
        """
        Resets the perturbations.
        """
        # Clear the shared class-level thruster mask so failures do not
        # persist across epochs/evaluations when perturbation objects are rebuilt.
        Perturbation.thruster_mask = None

        perturbation_keys = self.next_rng_keys(5)
        self.perturbations = PerturbationList(
            [
                StuckOffThrusters(
                    self.env_cfg, self.model_cfg, perturbation_keys[0]
                ),  # 0
                StuckOnThrusters(
                    self.env_cfg, self.model_cfg, perturbation_keys[1]
                ),  # 1
                FaultyValve(self.env_cfg, self.model_cfg, perturbation_keys[2]),  # 2
                SaturatedThrust(
                    self.env_cfg, self.model_cfg, perturbation_keys[3]
                ),  # 3
                ThrustInstability(
                    self.env_cfg, self.model_cfg, perturbation_keys[4]
                ),  # 4
            ]
        )
        self._refresh_effect_states()
        if hasattr(self, "_state"):
            self._state = self._state.replace(
                rng=self._rng,
                perturbation_states=self.perturbation_states
            )

    def reset_disturbances(self) -> None:
        """
        Resets external disturbances so curriculum fractions stay fixed per epoch.
        """
        disturbance_key = self.next_rng_keys(1)[0]
        self.disturbances = DisturbanceList(
            [ConstantForceDisturbance(self.env_cfg, disturbance_key)]
        )
        self._refresh_effect_states()
        if hasattr(self, "_state"):
            self._state = self._state.replace(
                rng=self._rng,
                disturbance_states=self.disturbance_states,
            )
