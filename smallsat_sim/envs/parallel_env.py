from smallsat_sim.envs.base_env import BaseEnv

class ParallelEnv(BaseEnv):
    def __init__(self, args) -> None:
        super().__init__(args)