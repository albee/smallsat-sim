from smallsat_sim.envs.vec_env import VecEnv
from smallsat_sim.envs.perturbations import StuckOffThrusters, PerturbationList


class AstrobeeEnvVectorized(VecEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee_rl", model_name="astrobee"
        )
        super().__init__(args=args)
