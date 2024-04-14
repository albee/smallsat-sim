import numpy as np

class BasePlanner(object):
    """
    Base class of planner objects.
    """
    def __init__(self) -> None:
        pass

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns a reference point based on current observations
        """
        pass