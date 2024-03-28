from smallsat_sim.envs.base_env import BaseEnv

class BaselineEnv(BaseEnv):
    def __init__(self) -> None:
        super().__init__()
        
        self._load_cfg("baseline")