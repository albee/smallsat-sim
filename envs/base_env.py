import numpy as np

class BaseEnv(object):
    def __init__(self) -> None:
        pass

    def reset(self) -> None:
        """
        Resets environment to a desired state.
        """
        pass

    def step(self) -> None:
        """
        Simulate environment for one timestep.
        """
        pass

    def get_obs(self) -> np.array:
        """
        Return all states and optionally rewards
        """
        pass

    def _create_viewer(self) -> None:
        """
        Creates a viewer to visualize simulation
        """

    # Rewards should be in here as well