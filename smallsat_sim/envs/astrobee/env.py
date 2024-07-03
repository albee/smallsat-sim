from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.perturbations import StuckOffThrusters, PerturbationList


class AstrobeeEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee", model_name="astrobee"
        )
        super().__init__(args=args)

        # Instantiate perturbations
        self.perturbations = PerturbationList([StuckOffThrusters(self.model_cfg).stuck_off_thruster(1)])
