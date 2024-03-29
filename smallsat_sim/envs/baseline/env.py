from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim import SMALLSAT_SIM_LIB_DIR

import mujoco
import os

class BaselineEnv(BaseEnv):
    def __init__(self) -> None:
        super().__init__()
        self.cfg = self._load_cfg("baseline")
        self._init_mujoco()

    def _init_mujoco(self) -> None:
        smallsat = self.cfg['smallsat']['name']
        xml = os.path.join(SMALLSAT_SIM_LIB_DIR, smallsat, smallsat + ".xml")
        self.model = mujoco.MjModel.from_xml_path(xml)
        self.data = mujoco.MjData(self.model)
        self._create_viewer()

    def step(self) -> None:
        # Step simulation
        # Controller callback is called internally to retrieve inputs
        # See: https://mujoco.readthedocs.io/en/latest/APIreference/APIglobals.html#mjcb-control
        mujoco.mj_step(self.model,self.data)

        # Update renderer
        self.viewer.sync()