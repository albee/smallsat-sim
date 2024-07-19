from abc import ABC, abstractmethod
import numpy as np
from enum import Enum


class PerturbationStatus(Enum):
    """
    Document type of active perturbation.
    0 := Thruster fully operational
    1 := Thruster is stuck off
    2 := Thruster is stuck on
    3 := Thruster is experiencing a different failure (sampled from a GP)
    """
    OPERATIONAL = 0
    STUCK_OFF = 1
    STUCK_ON = 2
    SAMPLE_PERTURBATION = 3


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

        # Thruster mask to document the operational status of thrusters
        self.thruster_mask = np.full(self.nu, PerturbationStatus.OPERATIONAL)

    def select_thruster(self, index) -> int:
        """
        Return index if one is provided, randomly select one of the working thrusters otherwise.
        """
        if index:
            return index
        else:
            working_thrusters = np.where(self.thruster_mask == PerturbationStatus.OPERATIONAL)[0]
            random_index = np.random.choice(working_thrusters)
            return random_index

    @abstractmethod
    def apply(self, input: np.ndarray, current_sim_time=None) -> np.ndarray:
        pass

    @abstractmethod
    def key_callback(self, keycode=None) -> None:
        pass


class PerturbationList(object):
    """
    Applies multiple perturbations.
    """
    def __init__(self, perturbations: list[Perturbation]) -> None:
        super().__init__()
        # Save perturbations in array
        self.perturbations = perturbations
        
        # Initialize look-up dictionary for keycodes and perturbations
        self.keycode_dict = {
            " ": {
                "description": "stuck_off",
                "type": 1,
                "warning": "Could not stuck off thruster. "
                           "No StuckOffThruster Perturbation module defined.",
            },
            "=": {
                "description": "stuck_on",
                "type": 2,
                "warning": "Could not stuck on thruster. "
                           "No StuckOnThruster Perturbation module defined."
            },
            ";": {
                "description": "sample_perturbation",
                "type": 3,
                "warning": "Could not fail thruster. "
                           "No SamplePerturbation Perturbation module defined."
            }
        }

    def apply(self, input: np.ndarray, current_sim_time=None) -> np.ndarray:
        for perturbation in self.perturbations:
            input = perturbation.apply(input, current_sim_time)

        return input
    
    def reset_thruster(self, index) -> None:
        """
        Reset thruster.
        """
        self.perturbations[0].thruster_mask[index] = 0 # It doesn't matter which perturbation is used for the reset
        print(f"Thruster {index} is fully functional.")

    def key_callback(self, keycode):
        """
        Handles external keycall back calls based on registered perturbation modules inside of self.perturbations.
        """
        # Extract location of desired perturbation (if it exists)
        perturbation_index = self._check_registered_perturbations(keycode)

        # Apply the callback function
        if isinstance(perturbation_index, int):
            self.perturbations[perturbation_index].key_callback(keycode)

    def _check_registered_perturbations(self, keycode) -> None | int:
        # Iterate over perturbations to find a matching type
        for idx, perturbation in enumerate(self.perturbations):
            if perturbation.failure_type.value == self.keycode_dict[chr(keycode)]["type"]:
                return idx

        # Print warning if no matching perturbation is found and return None
        print(self.keycode_dict.get(chr(keycode), {}).get("warning", "No perturbation found for keycode."))
        return None
    
    def _get_perturbation_index_from_string(self, desired_perturbation: str) -> None | int:
        # Iterate over perturbations to find a matching type
        for idx, perturbation in enumerate(self.perturbations):
            if perturbation.failure_type == desired_perturbation:
                return idx

        # Print warning if no matching perturbation is found and return None
        print("warning", "No perturbation found for keycode.")
        return None


class StuckOffThrusters(Perturbation):
    """
    This perturbation completly shuts off/fails thrusters.
    """
    def __init__(self, model_config) -> None:
        super().__init__(model_config)

        self.failure_type = PerturbationStatus.STUCK_OFF

    def apply(self, input: np.ndarray, current_sim_time=None) -> np.ndarray:
        input[self.thruster_mask == PerturbationStatus.STUCK_OFF] = 0.0

        return input

    def stuck_off_thruster(self, index=None) -> None:
        """
        Method to shut off a random thruster or a specific one if provided.
        """
        thruster_index = self.select_thruster(index)
        self.thruster_mask[thruster_index] = PerturbationStatus.STUCK_OFF
        print(f"Thruster {thruster_index} is stuck off.")

    def key_callback(self, keycode=None) -> None:
        # Call correct method for key callbacks
        self.stuck_off_thruster(index=None)


class StuckOnThrusters(Perturbation):
    """
    Thrusters unable to be turned off.
    """
    def __init__(self, model_config) -> None:
        super().__init__(model_config)

        self.model_config = model_config

        self.failure_type = PerturbationStatus.STUCK_ON

    def apply(self, input: np.ndarray, current_sim_time=None) -> np.ndarray:
        # Find where to apply the perturbation
        stuck_on_thrusters_indices = np.where(self.thruster_mask == PerturbationStatus.STUCK_ON)[0]

        # Apply the perturbation
        for idx in stuck_on_thrusters_indices:
            input[idx] = self.model_config.Thrusters.thruster_list[idx].forcerange[1] # Max. thruster force

        return input
    
    def stuck_on_thruster(self, index=None) -> None:
        """
        Method to unable a random thruster or a specific one if provided, to shut off.
        """
        thruster_index = self.select_thruster(index)
        self.thruster_mask[thruster_index] = PerturbationStatus.STUCK_ON
        print(f"Thruster {thruster_index} is stuck on.")
    
    def key_callback(self, keycode=None) -> None:
        # Call correct method for key callbacks
        self.stuck_on_thruster(index=None)


class SamplePerturbation(Perturbation):
    """
    This is a prototype of a non-parametric, time-varying perturbation.
    """
    def __init__(self, model_config, max_duration) -> None:
        super().__init__(model_config)

        self.model_config = model_config

        self.failure_type = PerturbationStatus.SAMPLE_PERTURBATION

        # Control frequency
        self.control_frequency = 50 # in Hz

        # Upper bound on the duration of the perturbation
        self.max_duration = max_duration

        # Thruster mask to document when a perturbation has started
        self.ongoing_perturbation_mask = np.zeros(self.nu)

        # Number of samples in an input trajectory
        self.num_samples = 500

        # Matrix to save input trajectories sampled from the GP
        self.input_trajectories = np.zeros((self.nu, self.num_samples))

    def apply(self, input: np.ndarray, current_sim_time) -> np.ndarray:
        # Find where to apply the perturbation
        failing_thrusters_indices = np.where(self.thruster_mask == PerturbationStatus.SAMPLE_PERTURBATION)[0]

        for idx in failing_thrusters_indices:
            # Update the thruster input
            u_nominal = input[idx]
            input[idx] = self.get_perturbed_input(u_nominal, idx, current_sim_time)
            
            # Clip the sample to the thruster range
            input[idx] = np.clip(input[idx], 0, self.model_config.Thrusters.thruster_list[idx].forcerange[1])

        return input
    
    def sample_perturbation(self, index=None) -> None:
        """
        Method to fail a random thruster or a specific one if provided.
        """
        thruster_index = self.select_thruster(index)
        self.thruster_mask[thruster_index] = PerturbationStatus.SAMPLE_PERTURBATION
        print(f"Thruster {thruster_index} is failing.")

    def key_callback(self, keycode=None) -> None:
        # Call correct method for key callbacks
        self.sample_perturbation(index=None)

    def get_perturbed_input(self, u_nominal, idx, current_sim_time) -> float:
        """
        Get the perturbed thruster input from the trajectory sampled from the GP (at the right time).
        """
        # If the failure is starting now
        if self.ongoing_perturbation_mask[idx] == 0:
            self.input_trajectories[idx, :] = self.sample_input_trajectory(u_nominal)
            self.ongoing_perturbation_mask[idx] = current_sim_time
            return self.input_trajectories[idx, 0]
        # If the failure took place already but is still unraveling
        else:
            return self.input_trajectories[idx, int(np.floor(current_sim_time - self.ongoing_perturbation_mask[idx]))]
    

    def sample_input_trajectory(self, u_nominal) -> np.ndarray:
        """
        Sample the perturbed inputs from a GP.
        """
        # Time points
        t = np.linspace(0, self.max_duration, self.num_samples).reshape(-1, 1)

        # Define the mean vector and covariance matrix
        mu = self.mean_func(t, u_nominal).ravel()
        cov = self.rbf_kernel(t, t, u_nominal)

        # Sample a trajectory from the GP
        samples = np.random.multivariate_normal(mu, cov, 1)

        return samples
    
    # Mean function of the GP
    def mean_func(self, t: np.ndarray, u_nominal: float):
        b = 0.2
        c = 1.0
        return u_nominal * np.exp(-b * t) * np.cos(c * t)

    # RBF kernel
    def rbf_kernel(self, t1, t2, u_nominal, base_length_scale=0.1, sigma_f=0.1):
        length_scale = base_length_scale * u_nominal
        sqdist = np.sum(t1**2, 1).reshape(-1, 1) + np.sum(t2**2, 1) - 2 * np.dot(t1, t2.T)
        return sigma_f**2 * np.exp(-0.5 / length_scale**2 * sqdist)
