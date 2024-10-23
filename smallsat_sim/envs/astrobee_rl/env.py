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
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee_rl", model_name="astrobee"
        )
        super().__init__(args=args)

        # Instantiate perturbations
        self.perturbations = PerturbationList(
            [
                StuckOffThrusters(self.env_cfg, self.model_cfg),  # 0
                StuckOnThrusters(self.env_cfg, self.model_cfg),  # 1
                FaultyValve(self.env_cfg, self.model_cfg),  # 2
                SaturatedThrust(self.env_cfg, self.model_cfg),  # 3
                ThrustInstability(self.env_cfg, self.model_cfg), # 4
            ]
        )

        self.perturbations.perturbations[2].register_perturbation()

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
                ThrustInstability(self.env_cfg, self.model_cfg), # 4
            ]
        )
