from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim import SMALLSAT_SIM_LIB_DIR

import mujoco
import os

class AstrobeeEnv(BaseEnv):
    def __init__(self,args) -> None:
        # Load necessary config files
        self.env_cfg, self.lib_cfg = self._load_cfg(env_name="astrobee",
                                                    lib_name="astrobee")
        super().__init__(args=args)

    def _init_mujoco(self) -> None:
        smallsat = self.cfg['smallsat']['name']
        xml = os.path.join(SMALLSAT_SIM_LIB_DIR, smallsat, smallsat + ".xml")
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self._create_viewer()