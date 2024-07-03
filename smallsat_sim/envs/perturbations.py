from abc import ABC, abstractmethod
import numpy as np


class Perturbation(ABC):
    """
    Base class for all perturbations applied to the model.
    Perturbations are defined as changes to the model's
    dynamics such as a mismatch between desired thrust versus
    actual thrust.
    """
    def __init__(self, model_config) -> None:
        # Extract the number of thrusters
        self.nu = model_config.Thrusters.n_thrusters

    @abstractmethod
    def apply(self, input: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def key_callback() -> None:
        pass


class PerturbationList(Perturbation):
    """
    Applies multiple perturbations.
    """
    def __init__(self, perturbations: list[Perturbation]) -> None:
        super().__init__()
        # Save perturbations in array
        self.perturbations = perturbations

        # Initialize look-up dictionary for keycodes and disturbances
        self.keycode_dict = {
            " ": {
                "type": StuckOffThrusters,
                "warning": "Could not stuck off thruster. No StuckOffThruster Perturbation module defined.",
            }
        }

    def apply(self, input: np.ndarray) -> np.ndarray:
        for perturbation in self.perturbations:
            input = perturbation.apply(input)

        return input

    def key_callback(self, keycode):
        """
        Handles external keycall back calls based on registered perturbation
        modules inside of self.perturbations
        """
        # Extract location of desired perturbation (if it exists)
        perturbation_index = self._check_registered_perturbations(keycode)

        # Apply the callback function
        if isinstance(perturbation_index, int):
            self.perturbations[perturbation_index].key_callback()

    def _check_registered_perturbations(self, keycode) -> None | int:
        # Iterate over perturbations to find a matching type
        for idx, perturbation in enumerate(self.perturbations):
            if isinstance(perturbation, self.keycode_dict[chr(keycode)]["type"]):
                return idx

        # Print warning if no matching perturbation is found and return None
        print(self.keycode_dict[chr(keycode)]["warning"])
        return None


class StuckOffThrusters(Perturbation):
    """
    This perturbation completly shuts off/fails thrusters.
    """

    def __init__(self, model_config) -> None:
        super().__init__(model_config)

        # Create a thruster mask to document the operational status of thrusters
        # 1 := Thruster fully operational
        # 0 := Thruster is stuck off
        self.thruster_mask = np.ones(12)

    def apply(self, input: np.ndarray) -> np.ndarray:

        input[self.thruster_mask == 0] = 0.0

        return input

    def stuck_off_thruster(self, index=None) -> None:
        """
        Method to shut off a random thruster or a specific one if provided.
        """
        if index:
            self.thruster_mask[index] = 0
        else:
            indices_with_ones = np.where(self.thruster_mask == 1)[0]
            random_index = np.random.choice(indices_with_ones)
            print(f"Thruster {random_index} is stuck off.")
            self.thruster_mask[random_index] = 0

    def key_callback(self, keycode=None) -> None:

        # Call correct method for key callbacks
        self.stuck_off_thruster(index=None)


class StuckOnThrusters(Perturbation):
    """
    Thrusters unable to be turned off.
    """
    def __init__(self, model_config) -> None:
        super().__init__(model_config)

        # Create a thruster mask to document the operational status of thrusters
        self.thruster_mask = np.ones(12)

    def apply(self, input: np.ndarray) -> np.ndarray:

        input[self.thruster_mask == 0] = np.inf
        
        return input
    
    def stuck_on_thruster(self, index=None) -> None:
        if index:
            self.thruster_mask[index] = 0
        else:
            indices_with_ones = np.where(self.thruster_mask == 1)[0]
            random_index = np.random.choice(indices_with_ones)
            print(f"Thruster {random_index} is stuck off.")
            self.thruster_mask[random_index] = 0
    
    def key_callback(self, keycode=None) -> None:

        # Call correct method for key callbacks
        self.stuck_on_thruster(index=None)


class SamplePerturbation(Perturbation):
    """
    This is a prototype of a non-parametric, time-varying perturbation.
    """
    def __init__(self, model_config) -> None:
        super().__init__(model_config)

    def apply(self, input: np.ndarray) -> np.ndarray:
        return super().apply(input)
    

    
    def key_callback() -> None:
        pass