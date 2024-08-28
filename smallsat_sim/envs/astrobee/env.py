import numpy as np

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.perturbations import PerturbationList, StuckOffThrusters, StuckOnThrusters, SamplePerturbation
from smallsat_sim.envs.disturbances import DisturbanceList, ConstantForceDisturbance


class AstrobeeEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee", model_name="astrobee"
        )
        super().__init__(args=args)

        # Instantiate perturbations
        self.perturbations = PerturbationList([StuckOffThrusters(self.model_cfg), StuckOnThrusters(self.model_cfg), SamplePerturbation(self.model_cfg, 10.0)])

        # Instantiate disturbances
        #self.disturbances = DisturbanceList([ConstantForceDisturbance(0.05, np.array([1, 1, 1]))])

        # Fail Thruster 0
        self.perturbations.perturbations[0].stuck_off_thruster(1)
        self.perturbations.perturbations[0].stuck_off_thruster(3)
