from abc import ABC, abstractmethod
import numpy as np
import matplotlib.pyplot as plt


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

        # Create a thruster mask to document the operational status of thrusters
        # 1 := Thruster fully operational
        # 0 := Thruster is failing
        self.thruster_mask = np.ones(12)

        # Monitor which perturbations are active
        self.perturbation_status = np.empty(self.nu, dtype=str)
        self.perturbation_status[:] = 'none'

    def select_thruster(self, index=None) -> int:
        """
        Return the index of which thruster to fail and modify the mask accordingly. 
        Random choice if no index is provided.
        """
        if index:
            self.thruster_mask[index] = 0
            return index
        else:
            indices_with_ones = np.where(self.thruster_mask == 1)[0]
            random_index = np.random.choice(indices_with_ones)
            self.thruster_mask[random_index] = 0
            return random_index

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
                "type": "stuck_off",
                "warning": "Could not stuck off thruster."
                           "No StuckOffThruster Perturbation module defined.",
            },
            "=": {
                "type": "stuck_on",
                "warning": "Could not stuck on thruster. "
                           "No StuckOnThruster Perturbation module defined."
            }
        }

    def apply(self, input: np.ndarray) -> np.ndarray:
        for perturbation in self.perturbations:
            input = perturbation.apply(input)

        return input

    def key_callback(self, keycode):
        """
        Handles external keycall back calls based on registered perturbation modules inside of self.perturbations.
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

    def apply(self, input: np.ndarray) -> np.ndarray:
        input[self.thruster_mask == 0] = 0.0

        return input

    def stuck_off_thruster(self, index=None) -> None:
        """
        Method to shut off a random thruster or a specific one if provided.
        """
        idx = self.select_thruster(index=None)
        print(f"Thruster {idx} is stuck off.")
        self.perturbation_status[idx] = 'stuck_off'

    def key_callback(self, keycode=None) -> None:
        # Call correct method for key callbacks
        self.stuck_off_thruster(index=None)


class StuckOnThrusters(Perturbation):
    """
    Thrusters unable to be turned off.
    """
    def __init__(self, model_config) -> None:
        super().__init__(model_config)

    def apply(self, input: np.ndarray) -> np.ndarray:
        input[self.thruster_mask == 0] = np.inf

        return input
    
    def stuck_on_thruster(self, index=None) -> None:
        """
        Method to unable a random thruster or a specific one if provided, to shut off.
        """
        idx = self.select_thruster(index=None)
        print(f"Thruster {idx} is stuck on.")
        self.perturbation_status[idx] = 'stuck_on'
    
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
        input[self.thruster_mask == 0] = self.get_perturbed_inputs()
        return input
    
    def sample_perturbation(self, index=None) -> None:
        """
        Method to fail a random thruster or a specific one if provided.
        """
        idx = self.select_thruster(index=None)
        print(f"Thruster {idx} has failed.")
        self.perturbation_status[idx] = 'sample_perturbation'

    def key_callback(self, keycode=None) -> None:
        # Call correct method for key callbacks
        self.sample_perturbation(index=None)

    def get_perturbed_inputs(self):
        """
        Sample the perturbed inputs from a GP.
        """
        raise NotImplementedError
    
    # Mean function of the GP
    def mean_function(t, u_nominal):
        b = 0.2
        c = 1.0
        return u_nominal * np.exp(-b * t) * np.cos(c * t)

    # RBF kernel
    def rbf_kernel(t1, t2, u_nominal, base_length_scale=0.1, sigma_f=0.1):
        length_scale = base_length_scale * u_nominal
        sqdist = np.sum(t1**2, 1).reshape(-1, 1) + np.sum(t2**2, 1) - 2 * np.dot(t1, t2.T)
        return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)
    
    # Time points
    t = np.linspace(0, 1, 100).reshape(-1, 1)  # Time space # TODO: pass time upper bound as an argument

    # Define nominal input, mean vector and covariance matrix
    u_nominal = np.random.uniform(0, 1.0) # TODO: give actual input
    mu = mean_function(t, u_nominal).ravel()
    cov = rbf_kernel(t, t, u_nominal)

    # Sample multiple trajectories from the Gaussian process
    num_samples = 1 # TODO: number of zeros in the thruster mask
    samples = np.random.multivariate_normal(mu, cov, num_samples)

    # Clip the samples to the thruster range
    samples = np.clip(samples, 0, 0.7) # TODO: get force range from config
    