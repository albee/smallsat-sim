from abc import ABC, abstractmethod
import numpy as np

class Disturbance(ABC):
    """
    Base class for all disturbances applied to the model.
    Disturbances are defined as an external influence,
    which apply a force or torque on the model.
    """
    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply(self):
        pass

class DisturbanceList(Disturbance):
    """
    Applies multiple disturbances sequentially
    """
    def __init__(self, disturbances: list[Disturbance]) -> None:
        super().__init__()
        self.disturbances = disturbances

    def apply(self):
        pass

class ConstantForceDisturbance(Disturbance):
    """
    Applies a constant force disturbance in a specified direction
    """ 
    def __init__(self, magnitude: float, direction: np.ndarray) -> None:
        super().__init__()

        # Check dimensions
        if not np.isscalar(magnitude):
            raise ValueError("Magnitude must be a scalar")
        if direction.shape != (3,):
            raise ValueError("Direction must have shape (3,1)")

        self.magnitude = magnitude
        self.direction = direction / np.linalg.norm(direction)
        
        # Save constant disturbance as 6D array
        force, torque = self.magnitude * self.direction, np.array([0, 0, 0])
        self.disturbance = np.concatenate((force, torque))

    def apply(self) -> np.ndarray:
        return self.disturbance