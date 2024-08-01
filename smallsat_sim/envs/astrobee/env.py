import numpy as np

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.perturbations import (
    PerturbationList,
    StuckOffThrusters,
    StuckOnThrusters,
    SamplePerturbation,
)
from smallsat_sim.envs.disturbances import DisturbanceList, ConstantForceDisturbance


class AstrobeeEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee", model_name="astrobee"
        )
        super().__init__(args=args)

        # Instantiate perturbations
        self.perturbations = PerturbationList(
            [
                StuckOffThrusters(self.env_cfg, self.model_cfg),
                StuckOnThrusters(self.env_cfg, self.model_cfg),
                SamplePerturbation(self.env_cfg, self.model_cfg, 10.0),
            ]
        )

        # Instantiate disturbances
        self.disturbances = DisturbanceList([ConstantForceDisturbance(self.env_cfg)])
