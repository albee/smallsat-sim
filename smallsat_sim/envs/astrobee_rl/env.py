import jax.numpy as jnp

from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.envs.perturbations import (
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
        run_name: str | None = None,
        init_pos: jnp.ndarray | None = None,
        max_start_offset: float | None = None,
        use_pretrained: bool | None = None,
        use_adaptive_approach: bool | None = None,
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
        if use_pretrained is not None:
            self.env_cfg.control.RL.use_pretrained = use_pretrained
        if use_adaptive_approach is not None:
            self.env_cfg.control.RL.use_adaptive_approach = use_adaptive_approach
        super().__init__(args=args)

        # Instantiate perturbations
        self.perturbations = PerturbationList(
            [
                StuckOffThrusters(self.env_cfg, self.model_cfg),  # 0
                StuckOnThrusters(self.env_cfg, self.model_cfg),  # 1
                FaultyValve(self.env_cfg, self.model_cfg),  # 2
                SaturatedThrust(self.env_cfg, self.model_cfg),  # 3
                ThrustInstability(self.env_cfg, self.model_cfg),  # 4
            ]
        )

        # self.perturbations.perturbations[2].register_perturbation()

        # Instantiate disturbances
        self.disturbances = DisturbanceList([ConstantForceDisturbance(self.env_cfg)])

    def reset_perturbations(self) -> None:
        """
        Resets the perturbations
        """
        self.perturbations = PerturbationList(
            [
                StuckOffThrusters(self.env_cfg, self.model_cfg),  # 0
                StuckOnThrusters(self.env_cfg, self.model_cfg),  # 1
                FaultyValve(self.env_cfg, self.model_cfg),  # 2
                SaturatedThrust(self.env_cfg, self.model_cfg),  # 3
                ThrustInstability(self.env_cfg, self.model_cfg),  # 4
            ]
        )
