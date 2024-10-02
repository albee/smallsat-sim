import numpy as np

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.perturbations import PerturbationList, StuckOffThrusters, StuckOnThrusters, SamplePerturbation, FaultyValve, SaturatedThrust
from smallsat_sim.envs.disturbances import DisturbanceList, ConstantForceDisturbance


class AstrobeeEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee", model_name="astrobee"
        )
        super().__init__(args=args)

        # Instantiate perturbations
        self.perturbations = PerturbationList([StuckOffThrusters(self.model_cfg), StuckOnThrusters(self.model_cfg), SamplePerturbation(self.model_cfg, 10.0), FaultyValve(self.model_cfg), SaturatedThrust(self.model_cfg)])

        # Instantiate disturbances
        #self.disturbances = DisturbanceList([ConstantForceDisturbance(0.05, np.array([1, 1, 1]))])

        # Fail Thruster 0
        #self.perturbations.perturbations[0].stuck_off_thruster(0, 70.0)
        #self.perturbations.perturbations[0].stuck_off_thruster(2, 70.0)
        
        # self.perturbations.perturbations[3].register_perturbation(0, 0.0, 0.15)
        #self.perturbations.perturbations[3].register_perturbation(2, 0.0, 0.0)

        
        #self.perturbations.perturbations[3].register_perturbation(5, 0.0, 0.0)
        #self.perturbations.perturbations[4].register_perturbation(7, 0.0, 0.0)

        #self.perturbations.perturbations[4].register_perturbation(9, 0.0, 0.0)
        #self.perturbations.perturbations[4].register_perturbation(1, 0.0, 0.0)