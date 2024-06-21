from smallsat_sim.envs.parallel_env import ParallelEnv
from smallsat_sim.envs.perturbations import StuckOffThrusters, PerturbationList


class AstrobeeEnvParallel(ParallelEnv):
    def __init__(self, args) -> None:
        # Load necessary config files
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee_parallel", model_name="astrobee"
        )
        super().__init__(args=args)
