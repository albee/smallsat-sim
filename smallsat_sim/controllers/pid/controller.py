from smallsat_sim.controllers.base_controller import BaseController

class PIDController(BaseController):
    def __init__(self) -> None:
        super().__init__()

        # Load specifc parameters from cfg folder and assign to self