from smallsat_sim.controllers.base_controller import BaseController

import numpy as np
import mujoco

class OpenLoopController(BaseController):
    def __init__(self) -> None:
        super().__init__()

    def get_control_input(self, env) -> None:
        """
        Calculates the open-loop control input
        """
        force0 = np.cos(env.data.time)
        force1 = np.cos(env.data.time + np.pi)

        return np.array([force0, force1])