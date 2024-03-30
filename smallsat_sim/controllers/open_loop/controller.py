from smallsat_sim.controllers.controller import BaseController

import numpy as np
import mujoco

class OpenLoopController(BaseController):
    def __init__(self) -> None:
        super().__init__()
        
        # register controller callback in mujoco
        mujoco.set_mjcb_control(self.controller_callback)

    def controller_callback(self, model, data) -> None:
        force0 = np.cos(data.time)
        force1 = np.cos(data.time + np.pi)

        data.ctrl[0] = force0
        data.ctrl[1] = force1