from controllers.base_controller import BaseController

class PID(BaseController):
    def __init__(self) -> None:
        super().__init__()

        # Load specifc parameters from cfg folder and assign to self