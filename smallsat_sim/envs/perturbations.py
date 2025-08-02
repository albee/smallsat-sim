from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List
import numpy as np
import jax
import jax.numpy as jnp

from smallsat_sim.envs.base_env_config import BaseEnvConfig
from smallsat_sim.model.base_model_config import BaseModelConfig
import matplotlib.pyplot as plt
import gpytorch
import torch
import warnings
from enum import Enum
from copy import deepcopy

# Import for typing
from smallsat_sim.model.base_model_config import BaseModelConfig

from scipy.interpolate import interp1d


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
    FAULTY_VALVE = 4
    SATURATED_THRUST = 5
    THRUST_INSTABILITY = 6


class Perturbation(ABC):
    """
    Base class for all perturbations applied to the model.
    Perturbations are defined as changes to the model's
    dynamics such as a mismatch between desired thrust versus
    actual thrust.
    """

    def __init__(
        self, env_config: BaseEnvConfig, model_config: BaseModelConfig
    ) -> None:
        # Extract the number of thrusters
        self.nu = model_config.Thrusters.n_thrusters

        # Find out how many environments there are
        if hasattr(env_config.control, "RL"):
            self.num_envs = env_config.control.RL.num_envs
        else:
            self.num_envs = 1

        # Thruster env_mask to document the operational status of thrusters
        self.thruster_mask = jnp.full(
            (self.num_envs, self.nu), PerturbationStatus.OPERATIONAL.value
        )

        # Vectorized thruster selection
        self.select_thrusters = jax.vmap(self.select_thruster)

    def select_thruster(
        self,
        thruster_mask_row: jnp.ndarray,
        perturbation_status: int,
        index: Optional[int] = None,
    ) -> jnp.ndarray:
        """
        Return updated row of the thruster mask. Use index if one is provided, randomly select one of the working thrusters (in each env) otherwise.
        """
        if isinstance(index, int):
            return thruster_mask_row.at[index].set(perturbation_status)
        else:
            working_thrusters = jnp.where(
                self.thruster_mask[0, :] == PerturbationStatus.OPERATIONAL.value
            )[0]
            key = jax.random.PRNGKey(np.random.randint(0, 99))  # No reproducability
            random_index = jax.random.choice(key, working_thrusters)
            return thruster_mask_row.at[random_index].set(perturbation_status)

    def get_perturbed_envs(
        self, frac_envs: float, perturbed_envs: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Return array with envs where a failure occurs.
        """
        if perturbed_envs is not None:
            return perturbed_envs
        else:
            fraction_perturbed_envs = frac_envs
            key = jax.random.PRNGKey(np.random.randint(0, 99))  # No reproducability
            return jax.random.randint(
                key,
                shape=(int(fraction_perturbed_envs * self.num_envs),),
                minval=0,
                maxval=self.num_envs,
            )

    @abstractmethod
    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = 0.0
    ) -> jnp.ndarray:
        pass

    @abstractmethod
    def key_callback(self, keycode: Optional[int] = None) -> None:
        pass


class PerturbationList(ABC):
    """
    Applies multiple perturbations.
    """

    def __init__(self, perturbations: List[Perturbation]) -> None:
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
                "No StuckOnThruster Perturbation module defined.",
            },
            ";": {
                "description": "sample_perturbation",
                "type": 3,
                "warning": "Could not fail thruster. "
                "No SamplePerturbation Perturbation module defined.",
            },
        }

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = 0.0
    ) -> jnp.ndarray:
        for perturbation in self.perturbations:
            input = perturbation.apply(input, timestamp)

        return input

    def reset_thruster(self, index: int) -> None:
        """
        Reset thruster.
        """
        self.perturbations[0].thruster_mask = (
            self.perturbations[0].thruster_mask.at[:, index].set(0)
        )  # It doesn't matter which perturbation is used for the reset
        print(f"Thruster {index} is fully functional.")

    def key_callback(self, keycode: Optional[int] = None):
        """
        Handles external keycall back calls based on registered perturbation modules inside of self.perturbations.
        """
        # Extract location of desired perturbation (if it exists)
        perturbation_index = self._check_registered_perturbations(keycode)

        # Apply the callback function
        if isinstance(perturbation_index, int):
            self.perturbations[perturbation_index].key_callback(keycode)

    def _check_registered_perturbations(
        self, keycode: Optional[int] = None
    ) -> Optional[int]:
        # Iterate over perturbations to find a matching type
        for idx, perturbation in enumerate(self.perturbations):
            if (
                perturbation.failure_type.value
                == self.keycode_dict[chr(keycode)]["type"]
            ):
                return idx

        # Print warning if no matching perturbation is found and return None
        print(
            self.keycode_dict.get(chr(keycode), {}).get(
                "warning", "No perturbation found for keycode."
            )
        )
        return None

    def _get_perturbation_index_from_string(
        self, desired_perturbation: str
    ) -> Optional[int]:
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

    def __init__(
        self,
        env_config: BaseEnvConfig,
        model_config: BaseModelConfig,
        perturbed_envs: Optional[jnp.ndarray] = None,
    ) -> None:
        super().__init__(env_config, model_config)

        self.failure_type = PerturbationStatus.STUCK_OFF
        self.start_times = np.zeros(self.nu)

    def apply(self, input: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:

        # Do a elementwise AND operation
        stuck_off = np.logical_and(
            self.thruster_mask == PerturbationStatus.STUCK_OFF,
            timestamp >= self.start_times,
        )

        # Shut off respective thrusters
        input[stuck_off] = 0.0

        return input

    def stuck_off_thruster(
        self, index: Optional[int] = None, start_time: Optional[float] = 0.0
    ) -> None:
        """
        Method to shut off a random thruster or a specific one if provided.
        """
        thruster_index = self.select_thruster(index)
        self.thruster_mask[thruster_index] = PerturbationStatus.STUCK_OFF
        self.start_times[thruster_index] = start_time
        print(
            f"Thruster {thruster_index} is stuck off starting at {start_time} seconds."
        )

    def key_callback(self, keycode: Optional[int] = None) -> None:
        # Call correct method for key callbacks
        self.stuck_off_thruster()


class StuckOnThrusters(Perturbation):
    """
    Thrusters unable to be turned off.
    """

    def __init__(
        self,
        env_config: BaseEnvConfig,
        model_config: BaseModelConfig,
        perturbed_envs: Optional[jnp.ndarray] = None,
    ) -> None:
        super().__init__(env_config, model_config)

        self.failure_type = PerturbationStatus.STUCK_ON
        self.start_times = np.zeros(self.nu)

    def apply(self, input: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        # Find where to apply the perturbation
        stuck_on_thrusters_indices = np.where(
            np.logical_and(
                self.thruster_mask == PerturbationStatus.STUCK_ON,
                timestamp >= self.start_times,
            )
        )[0]

        self.perturbed_envs = perturbed_envs

        # Vectorized input replacement
        self.replace_thrust_inputs = jax.vmap(self.replace_with_upper_thrust_bound)

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = 0.0
    ) -> jnp.ndarray:
        input = input.reshape(
            self.num_envs, self.nu
        )  # Reshape to account for multiple envs
        if self.is_active and timestamp >= self.start_time:
            self.replace_thrust_inputs(input)

        return input

    def stuck_on_thruster(
        self, index: Optional[int] = None, start_time: Optional[float] = 0.0
    ) -> None:
        """
        Method to unable a random thruster or a specific one if provided, to shut off.
        """
        thruster_index = self.select_thruster(index)
        self.thruster_mask[thruster_index] = PerturbationStatus.STUCK_ON
        self.start_times[thruster_index] = start_time
        print(f"Thruster {thruster_index} is stuck on.")

    def key_callback(self, keycode: Optional[int] = None) -> None:
        # Call correct method for key callbacks
        self.stuck_on_thruster()

    def replace_with_upper_thrust_bound(self, input_row: jnp.ndarray) -> jnp.ndarray:
        """
        Return input row with updated thrust values.
        """
        # Apply the perturbation
        updated_input_row = input_row
        for i in range(len(input_row)):
            if input_row[i].val[0] == PerturbationStatus.STUCK_ON.value:
                updated_input_row = input_row.at[i].set(
                    self.model_config.Thrusters.thruster_list[i].forcerange[1]
                )  # Max. thruster force
        return updated_input_row


class SamplePerturbation(Perturbation):
    """
    This is a prototype of a non-parametric, time-varying perturbation.
    """

    def __init__(
        self,
        env_config: BaseEnvConfig,
        model_config: BaseModelConfig,
        max_duration,
        perturbed_envs: Optional[jnp.ndarray] = None,
    ) -> None:
        super().__init__(env_config, model_config)

        self.model_config = model_config

        self.failure_type = PerturbationStatus.SAMPLE_PERTURBATION

        self.is_active = False

        self.perturbed_envs = perturbed_envs

        # Control frequency
        self.control_frequency = 50  # in Hz

        # Upper bound on the duration of the perturbation
        self.max_duration = max_duration

        # Thruster mask to document when a perturbation has started
        self.ongoing_perturbation_mask = jnp.zeros(self.nu)

        # Number of samples in an input trajectory
        self.num_samples = 500

        # Matrix to save input trajectories sampled from the GP
        self.input_trajectories = jnp.zeros((self.nu, self.num_samples))

        # Vectorized input replacement
        self.replace_thrust_inputs = jax.vmap(self.replace_with_traj_sample)

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = 0.0
    ) -> jnp.ndarray:
        input = input.reshape(
            self.num_envs, self.nu
        )  # Reshape to account for multiple envs
        if self.is_active and timestamp >= self.start_time:
            input = self.replace_thrust_inputs(
                input, jnp.full(self.num_envs, timestamp)
            )

        return input

    def sample_perturbation(self, index: Optional[int] = None, start_time=0.0) -> None:
        """
        Method to fail a random thruster or a specific one if provided.
        """
        self.start_time = start_time
        self.is_active = True
        self.perturbed_envs = self.get_perturbed_envs(0.1)
        self.thruster_mask = self.thruster_mask.at[self.perturbed_envs, :].set(
            self.select_thrusters(
                self.thruster_mask[self.perturbed_envs, :],
                jnp.full(
                    self.perturbed_envs.shape[0],
                    PerturbationStatus.SAMPLE_PERTURBATION.value,
                ),
                index,
            )
        )
        print(f"Thruster(s) failing.")

    def key_callback(self, keycode: Optional[int] = None) -> None:
        # Call correct method for key callbacks
        self.sample_perturbation()

    def replace_with_traj_sample(
        self, input_row: jnp.ndarray, timestamp: float
    ) -> jnp.ndarray:
        """
        Return input row with updated thrust values.
        """
        # Apply the perturbation
        updated_input_row = input_row
        for i in range(len(input_row)):
            if input_row[i].val[0] == PerturbationStatus.SAMPLE_PERTURBATION.value:
                updated_input_row = input_row.at[i].set(
                    jnp.min(
                        self.get_perturbed_input(input_row[i].val[0], i, timestamp),
                        self.model_config.Thrusters.thruster_list[i].forcerange[1],
                    )
                )
        return updated_input_row

    def get_perturbed_input(
        self, u_nominal: float, idx: int, timestamp: float
    ) -> float:
        """
        Get the perturbed thruster input from the trajectory sampled from the GP (at the right time).
        """
        # If the failure is starting now
        if self.ongoing_perturbation_mask[idx] == 0:
            self.input_trajectories[idx, :] = self.sample_input_trajectory()
            self.ongoing_perturbation_mask[idx] = timestamp
            return self.input_trajectories[idx, 0]
        # If the failure took place already but is still unraveling
        else:
            return self.input_trajectories[
                idx,
                int(jnp.floor(timestamp - self.ongoing_perturbation_mask[idx])),
            ]

    def sample_input_trajectory(self, u_nominal: float = 1.0) -> jnp.ndarray:
        """
        Sample the perturbed inputs from a GP.
        """
        # Time points
        t = jnp.linspace(0, self.max_duration, self.num_samples).reshape(-1, 1)

        # Define the mean vector and covariance matrix
        mu = self.mean_func(t, u_nominal).ravel()
        cov = self.rbf_kernel(t, t, u_nominal)

        # Sample a trajectory from the GP
        key = jax.random.PRNGKey(np.random.randint(0, 99))  # No reproducability
        samples = jax.random.multivariate_normal(key, mu, cov, 1)

        return jnp.tile(samples, (self.num_envs, 1))

    # Mean function of the GP
    def mean_func(self, t: jnp.ndarray, u_nominal: float):
        b = 0.2
        c = 1.0
        return u_nominal * jnp.exp(-b * t) * jnp.cos(c * t)

    # RBF kernel
    def rbf_kernel(
        self,
        t1: jnp.ndarray,
        t2: jnp.ndarray,
        u_nominal: float = 1.0,
        base_length_scale: float = 0.1,
        sigma_f: float = 0.1,
    ):
        length_scale = base_length_scale * u_nominal
        sqdist = (
            jnp.sum(t1**2, 1).reshape(-1, 1) + jnp.sum(t2**2, 1) - 2 * jnp.dot(t1, t2.T)
        )
        return sigma_f**2 * jnp.exp(-0.5 / length_scale**2 * sqdist)


"""
This section will consist of "phyisically-grounded" perturbations modelled by a GP.
Note that these function take in the demanded force and output the actual force.

"""


class ThrusterFailureSimulator:
    """
    This class is used to the data for the nonlinar perturbations.
    It does the following things (in sequence):
        - 
    """
    def __init__(
        self,
        num_points=100,
        upper_bound=0.6,
        subset_size=40,
        valve_min=0.1,
        valve_max=0.5,
    ):
        self.num_points = num_points
        self.upper_bound = upper_bound
        self.subset_size = subset_size
        self.valve_min = valve_min
        self.valve_max = valve_max
        self.demanded_force = torch.linspace(
            0, upper_bound, num_points
        )  # Values between 0 and upper_bound
        self.failure_modes = self._get_failure_modes()

    class ThrusterFailureGPModel(gpytorch.models.ExactGP):
        def __init__(
            self,
            train_x,
            train_y,
            likelihood,
            kernel_type="RBF",
            lengthscale=0.2,
            outputscale=1.0,
        ):
            super().__init__(train_x, train_y, likelihood)
            self.mean_module = gpytorch.means.ConstantMean()
            self.covar_module = self._choose_kernel(kernel_type, lengthscale)
            self.covar_module.outputscale = outputscale

        def _choose_kernel(self, kernel_type, lengthscale):
            if kernel_type == "RBF":
                return gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.RBFKernel(lengthscale=lengthscale)
                )
            elif kernel_type == "Matern":
                return gpytorch.kernels.ScaleKernel(
                    gpytorch.kernels.MaternKernel(nu=1.5, lengthscale=lengthscale)
                )
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        def forward(self, x):
            mean_x = self.mean_module(x)
            covar_x = self.covar_module(x)
            return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)

    def generate_failure_data(self, failure_type):
        actual_force = self._get_actual_force(failure_type)
        subset_indices = self._select_subset_indices()
        demanded_force_subset = self.demanded_force[subset_indices]
        actual_force_subset = actual_force[subset_indices]

        # Initialize GP model and likelihood
        params = self.failure_modes[failure_type]
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        likelihood.noise = torch.tensor(1e-4)

        model = self.ThrusterFailureGPModel(
            demanded_force_subset,
            actual_force_subset,
            likelihood,
            kernel_type=params["kernel"],
            lengthscale=params["lengthscale"],
            outputscale=params["outputscale"],
        )

        # No training needed as hyperparameters are manually set
        x_test = torch.linspace(0, self.upper_bound, self.num_points)
        sampled_function = self._sample_gp_function(model, likelihood, x_test, failure_type)

        sampled_function = torch.clamp(sampled_function, 0, self.upper_bound)

        return x_test, sampled_function

    def _get_actual_force(self, failure_type):
        if failure_type == PerturbationStatus.SATURATED_THRUST:
            return torch.where(
                self.demanded_force < 0.2 * self.upper_bound,
                self.demanded_force,
                0.2 * self.upper_bound
                + 0.0005
                * torch.randn_like(self.demanded_force)
                * (self.upper_bound - self.demanded_force),
            )
        elif failure_type == PerturbationStatus.FAULTY_VALVE:
            valve_min, valve_max = self.valve_min, self.valve_max
            lower_threshold = 0.67 * self.upper_bound
            upper_threshold = 0.73 * self.upper_bound
            actual_force = torch.full_like(self.demanded_force, valve_min)
            linear_region = (self.demanded_force >= lower_threshold) & (
                self.demanded_force <= upper_threshold
            )
            actual_force[linear_region] = valve_min + (
                (self.demanded_force[linear_region] - lower_threshold)
                / (upper_threshold - lower_threshold)
            ) * (valve_max - valve_min)
            actual_force[self.demanded_force > upper_threshold] = valve_max
            return actual_force
        elif failure_type == PerturbationStatus.THRUST_INSTABILITY:
            return (
                self.demanded_force
                - 0.1 * self.upper_bound * torch.sin(10 * self.demanded_force)
                + 0.05 * self.upper_bound * torch.randn_like(self.demanded_force)
            )
        elif failure_type == "thermal_stress":
            return self.demanded_force * torch.exp(
                -0.5 * (self.demanded_force / self.upper_bound)
            )
        else:
            raise ValueError(f"Unknown failure type: {failure_type}")

    def _select_subset_indices(self):
        indices = np.random.choice(
            range(1, self.num_points - 1), size=self.subset_size - 2, replace=False
        )
        indices = np.concatenate(([0], indices, [self.num_points - 1]))
        return np.sort(indices)

    def _sample_gp_function(self, gp_model, likelihood, demanded_force, failure_type):
        gp_model.eval()
        likelihood.eval()
        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            observed_pred = gp_model(demanded_force)

        if failure_type == PerturbationStatus.SATURATED_THRUST:
            return observed_pred.mean
        else:
            return observed_pred.sample()

    def _get_failure_modes(self):
        return {
            PerturbationStatus.SATURATED_THRUST: {
                "kernel": "Matern",
                "lengthscale": 0.1,
                "outputscale": 0.01,
            },
            PerturbationStatus.FAULTY_VALVE: {
                "kernel": "Matern",
                "lengthscale": 0.3,
                "outputscale": 0.4,
            },
            PerturbationStatus.THRUST_INSTABILITY: {
                "kernel": "Matern",
                "lengthscale": 0.15,
                "outputscale": 0.5,
            },
            "thermal_stress": {
                "kernel": "Matern",
                "lengthscale": 0.15,
                "outputscale": 0.5,
            },
        }


class GPPerturbation(Perturbation):
    def __init__(self, model_config, failure_type) -> None:
        super().__init__(model_config)

        self.failure_type = failure_type
        self.start_times = np.zeros(self.nu)
        self.thruster_list = deepcopy(model_config.Thrusters.thruster_list)

        # Store interpolation functions for each thruster
        self.interpolations = [None] * self.nu

    def apply(self, input: np.ndarray, timestamp: Optional[float] = None) -> np.ndarray:
        # Do a elementwise AND operation
        faulty = np.logical_and(
            self.thruster_mask == self.failure_type,
            timestamp >= self.start_times,
        )

        # Apply interpolation only to the affected thrusters with active perturbations
        if np.any(faulty):
            # Gather input values for the affected thrusters
            affected_inputs = input[faulty]

            # Apply the respective interpolation functions
            interpolated_values = np.array(
                [
                    self.interpolations[i](affected_inputs[idx])
                    for idx, i in enumerate(np.where(faulty)[0])
                ]
            )

            # Update the input array with interpolated values
            input[faulty] = interpolated_values

        return input

    def register_perturbation(
        self,
        index: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        valve_min: Optional[float] = None,
        valve_max: Optional[float] = None,
    ) -> None:
        """
        Register a perturbation and store the interpolation data.
        """

        thruster_index = self.select_thruster(index)
        self.thruster_mask[thruster_index] = self.failure_type
        self.start_times[thruster_index] = start_time

        if valve_min is None:
            valve_min = 0.15 * self.thruster_list[thruster_index].ctrlrange[-1]

        if valve_max is None:
            valve_max = 0.8 * self.thruster_list[thruster_index].ctrlrange[-1]

        # Get the GP-data
        x_data, y_data = ThrusterFailureSimulator(
            upper_bound=self.thruster_list[thruster_index].ctrlrange[-1],
            valve_min=valve_min,
            valve_max=valve_max,
        ).generate_failure_data(self.failure_type)

        # Plot data
        if False:
            plt.figure()
            plt.plot(x_data, y_data, "--")

            plt.show()

        # Create the interpolation function for the affected thruster
        self.interpolations[thruster_index] = interp1d(
            x_data, y_data, kind="linear", fill_value="extrapolate"
        )

        print(
            f"Thruster {thruster_index} is affected by a faulty valve starting at {start_time} seconds."
        )

    def key_callback(self, keycode: Optional[int] = None) -> None:
        pass


class FaultyValve(GPPerturbation):
    def __init__(self, model_config) -> None:
        super().__init__(model_config, PerturbationStatus.FAULTY_VALVE)

    def register_perturbation(
        self,
        index: int | None = None,
        start_time: float | None = 0,
        valve_min: float | None = None,
        valve_max: float | None = None,
    ) -> None:
        if valve_min is None or valve_max is None:
            warnings.warn(
                "No min or max value set for the FaultyValve perturbation.", UserWarning
            )
        return super().register_perturbation(index, start_time, valve_min, valve_max)


class SaturatedThrust(GPPerturbation):
    def __init__(self, model_config) -> None:
        super().__init__(model_config, PerturbationStatus.SATURATED_THRUST)

    def register_perturbation(
        self,
        index: int | None = None,
        start_time: float | None = 0,
        valve_min: float | None = None,
        valve_max: float | None = None,
    ) -> None:
        return super().register_perturbation(index, start_time, valve_min, valve_max)
    
class ThrustInstability(GPPerturbation):
    def __init__(self, model_config) -> None:
        super().__init__(model_config, PerturbationStatus.THRUST_INSTABILITY)

    def register_perturbation(
        self,
        index: int | None = None,
        start_time: float | None = 0,
        valve_min: float | None = None,
        valve_max: float | None = None,
    ) -> None:
        return super().register_perturbation(index, start_time, valve_min, valve_max)
