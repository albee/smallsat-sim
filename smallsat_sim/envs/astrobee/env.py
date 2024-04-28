from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.perturbations import StuckOffThrusters


class AstrobeeEnv(BaseEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee", model_name="astrobee"
        )
        super().__init__(args=args)

        # Instantiate perturbations
        self.perturbation = StuckOffThrusters(self.model_cfg)
