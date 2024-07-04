from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.perturbations import PerturbationList, StuckOffThrusters, StuckOnThrusters, SamplePerturbation


class AstrobeeEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee", model_name="astrobee"
        )
        super().__init__(args=args)

        # Instantiate perturbations
        self.perturbations = PerturbationList([StuckOffThrusters(self.model_cfg), StuckOnThrusters(self.model_cfg), SamplePerturbation(self.model_cfg)])
