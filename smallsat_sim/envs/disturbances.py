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
    def apply(self) -> np.ndarray:
        pass

    @abstractmethod
    def key_callback(self, keycode=None) -> None:
        pass


class DisturbanceList(object):
    """
    Applies multiple disturbances sequentially
    """
    def __init__(self, disturbances: list[Disturbance]) -> None:
        super().__init__()
        self.disturbances = disturbances

        # Initialize look-up dictionary for keycodes and disturbances
        self.keycode_dict = {
            "/": {
                "type": "const_force_disturbance",
                "warning": "Could not activate constant force disturbance. "
                           "No ConstantForceDisturbance module defined.",
            }
        }

    def apply(self):
        # Net force vector of all the disturbances in the list
        self.net_force = np.zeros(6)

        for disturbance in self.disturbances:
            self.net_force += disturbance.apply()

        return self.net_force
    
    def key_callback(self, keycode):
        """
        Handles external keycall back calls based on registered disturbance modules inside of self.disturbances.
        """
        # Extract location of desired disturbance (if it exists)
        disturbance_index = self._check_registered_disturbances(keycode)

        # Apply the callback function
        if isinstance(disturbance_index, int):
            self.disturbances[disturbance_index].key_callback(keycode)

    def _check_registered_disturbances(self, keycode) -> None | int:
        # Iterate over disturbances to find a matching type
        for idx, disturbance in enumerate(self.disturbances):
            if disturbance.failure_type == self.keycode_dict[chr(keycode)]["type"]:
                return idx

        # Print warning if no matching disturbance is found and return None
        print(self.keycode_dict.get(chr(keycode), {}).get("warning", "No disturbance found for keycode."))
        return None
    
    def _get_disturbance_index_from_string(self, desired_disturbance: str) -> None | int:
        # Iterate over disturbances to find a matching type
        for idx, disturbance in enumerate(self.disturbances):
            if disturbance.failure_type == desired_disturbance:
                return idx

        # Print warning if no matching perturbation is found and return None
        print("warning", "No perturbation found for keycode.")
        return None


class ConstantForceDisturbance(Disturbance):
    """
    Applies a constant force disturbance in a specified direction
    """ 
    def __init__(self, magnitude: float, direction: np.ndarray) -> None:
        super().__init__()

        self.failure_type = 'const_force_disturbance'

        self.is_active = False

        # Check dimensions
        if not np.isscalar(magnitude):
            raise ValueError("Magnitude must be a scalar")
        if direction.shape != (3,):
            raise ValueError("Direction must have shape (3,1)")

        self.magnitude = magnitude
        self.direction = direction / np.linalg.norm(direction)
        
        # Save constant disturbance as 6D array
        force, torque = self.magnitude * self.direction, np.array([0, 0, 0])
        self.const_force = np.concatenate((force, torque))

    def apply(self) -> np.ndarray:
        if not self.is_active:
            return np.zeros(6)
        
        return self.const_force
    
    def const_force_disturbance(self) -> None:
        """
        Activate the constant force disturbance.
        """
        self.is_active = True
        print("Constant force disturbance is active.")

    def deactivate_const_force_disturbance(self) -> None:
        """
        Reset disturbance.
        """
        self.is_active = False
        print("Constant force disturbance is inactive.")
    
    def key_callback(self, keycode=None) -> None:
        # Call correct method for key callbacks
        self.const_force_disturbance()
