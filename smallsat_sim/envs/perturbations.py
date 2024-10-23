from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List
import warnings
from copy import deepcopy
import numpy as np
import jax
import jax.numpy as jnp
from scipy.interpolate import interp1d
import gpflow
import tensorflow as tf
import tensorflow_probability as tfp

from smallsat_sim.envs.base_env_config import BaseEnvConfig
from smallsat_sim.model.base_model_config import BaseModelConfig

# Import for typing
from smallsat_sim.model.base_model_config import BaseModelConfig


class PerturbationStatus(Enum):
    """
    Document type of active perturbation.
    0 := Thruster fully operational
    1 := Thruster is stuck off
    2 := Thruster is stuck on
    3 := Thruster valve is faulty
    4 := Thruster output is saturated
    5 := Thruster output is unstable
    """

    OPERATIONAL = 0
    STUCK_OFF = 1
    STUCK_ON = 2
    FAULTY_VALVE = 3
    SATURATED_THRUST = 4
    THRUST_INSTABILITY = 5


class Perturbation(ABC):
    """
    Base class for all perturbations applied to the model.
    Perturbations are defined as changes to the model's
    dynamics such as a mismatch between desired thrust versus
    actual thrust.
    """

    thruster_mask = None  # Class attribute, shared by all instances

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
        if Perturbation.thruster_mask is None:  # Only initialize once
            Perturbation.thruster_mask = jnp.full(
                (self.num_envs, self.nu), PerturbationStatus.OPERATIONAL.value
            )

        # PRNG key to select among the working thrusters
        self.thruster_rng_key = jax.random.PRNGKey(42)

    def get_perturbed_envs(
        self, proportion: float, perturbed_envs: Optional[jnp.ndarray] = None
    ) -> jnp.ndarray:
        """
        Return array with envs where a failure occurs. If proportion and num_envs are too low, no envs will be perturbed.
        """
        if isinstance(perturbed_envs, jnp.ndarray):
            return perturbed_envs

        else:
            fraction_perturbed_envs = jnp.clip(proportion, 0.0, 1.0)
            num_perturbed_envs = int(fraction_perturbed_envs * self.num_envs)
            operational_thruster_mask = jnp.any(
                Perturbation.thruster_mask == PerturbationStatus.OPERATIONAL.value,
                axis=1,
            )  # Only select envs that still have functioning thrusters
            operational_envs = jnp.where(operational_thruster_mask)[0]
            key = jax.random.PRNGKey(np.random.randint(0, 9999))  # No reproducability
            shuffled_indices = jax.random.permutation(key, operational_envs)
            return shuffled_indices[:num_perturbed_envs]

    def select_thrusters(
        self,
        perturbed_envs: jnp.ndarray,
        perturbed_thrusters: Optional[jnp.ndarray] = None,
    ) -> jnp.ndarray:
        """
        Return thruster indices to fail.
        """

        if isinstance(perturbed_thrusters, jnp.ndarray):
            return perturbed_thrusters

        else:
            jax.random.split(self.thruster_rng_key, perturbed_envs.shape[0])

            def random_operational_thruster(key, row, max_size=self.nu):
                # Get indices of operational thrusters (pad with -1 if not enough)
                operational_thrusters = jnp.where(
                    row == PerturbationStatus.OPERATIONAL.value,
                    size=max_size,
                    fill_value=-1,
                )[0]

                # Replace invalid (-1) indices with a large number so they won't be selected
                valid_mask = operational_thrusters != -1
                masked_indices = jnp.where(valid_mask, operational_thrusters, max_size)

                # Randomly select a valid index
                random_idx = jax.random.choice(
                    key, masked_indices, p=(valid_mask / valid_mask.sum())
                )

                return random_idx

            # Select the rows from the array
            selected_envs = Perturbation.thruster_mask[perturbed_envs]

            # Generate subkeys for each row
            subkeys = jax.random.split(
                self.thruster_rng_key, num=perturbed_envs.shape[0]
            )

            # Vectorized selection of random zero indices
            vec_random_operational_thruster = jax.vmap(
                random_operational_thruster, in_axes=(0, 0)
            )
            selected_thrusters = vec_random_operational_thruster(subkeys, selected_envs)

            return selected_thrusters

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
        # NOTE: the keycodes correspond to the American keyboard layout
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
        }

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = 0.0
    ) -> jnp.ndarray:
        for perturbation in self.perturbations:
            input = perturbation.apply(input, timestamp)

        return input

    def reset_thrusters(
        self, perturbation_idx: int, envs: jnp.ndarray, thruster_indices: jnp.ndarray
    ) -> None:
        """
        Reset thrusters.
        """
        self.perturbations[perturbation_idx].thruster_mask = (
            self.perturbations[perturbation_idx]
            .thruster_mask.at[envs, thruster_indices]
            .set(PerturbationStatus.OPERATIONAL.value)
        )
        print(f"Reset thruster(s).")

    def key_callback(self, keycode: Optional[int] = None):
        """
        Handles external keycall back calls based on registered perturbation modules inside of self.perturbations.
        """
        # Extract location of desired perturbation (if it exists)
        perturbation_idx = self._check_registered_perturbations(keycode)

        # Apply the callback function
        if isinstance(perturbation_idx, int):
            self.perturbations[perturbation_idx].key_callback(keycode)

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
    ) -> None:
        super().__init__(env_config, model_config)

        self.failure_type = PerturbationStatus.STUCK_OFF
        self.start_times = jnp.zeros((self.num_envs, self.nu))

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = 0.0
    ) -> jnp.ndarray:
        input = input.reshape(
            self.num_envs, self.nu
        )  # Reshape to account for multiple envs

        input = input.at[
            jnp.logical_and(
                Perturbation.thruster_mask == PerturbationStatus.STUCK_OFF.value,
                timestamp >= self.start_times,
            )
        ].set(0.0)

        return input

    def stuck_off_thruster(
        self,
        perturbed_envs: Optional[jnp.ndarray] = None,
        perturbed_thrusters: Optional[jnp.ndarray] = None,
        start_time: Optional[float] = 0.0,
    ) -> None:
        """
        Method to shut off a random thruster or a specific one if provided.
        """
        stuck_off_envs = self.get_perturbed_envs(1.0, perturbed_envs)
        stuck_off_thrusters = self.select_thrusters(stuck_off_envs, perturbed_thrusters)

        Perturbation.thruster_mask = Perturbation.thruster_mask.at[
            stuck_off_envs, stuck_off_thrusters
        ].set(
            jnp.full(self.stuck_off_envs.shape[0], PerturbationStatus.STUCK_OFF.value)
        )

        self.start_times = self.start_times.at[stuck_off_envs, stuck_off_thrusters].set(
            start_time
        )

        print(f"Thruster(s) stuck off.")
        print("Thruster mask: ", Perturbation.thruster_mask)

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
    ) -> None:
        super().__init__(env_config, model_config)

        self.model_config = model_config

        self.failure_type = PerturbationStatus.STUCK_ON
        self.start_times = jnp.zeros((self.num_envs, self.nu))

        # Vectorized input replacement
        self.replace_thrust_inputs = jax.vmap(
            self.replace_with_upper_thrust_bound, in_axes=(0, 0, None)
        )

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = 0.0
    ) -> jnp.ndarray:
        input = input.reshape(
            self.num_envs, self.nu
        )  # Reshape to account for multiple envs

        self.replace_thrust_inputs(input, self.start_times, timestamp)

        return input

    def stuck_on_thruster(
        self,
        perturbed_envs: Optional[jnp.ndarray] = None,
        perturbed_thrusters: Optional[jnp.ndarray] = None,
        start_time: Optional[float] = 0.0,
    ) -> None:
        """
        Method to unable a random thruster or a specific one if provided, to shut off.
        """
        stuck_on_envs = self.get_perturbed_envs(1.0, perturbed_envs)

        stuck_on_thrusters = self.select_thrusters(stuck_on_envs, perturbed_thrusters)

        Perturbation.thruster_mask = Perturbation.thruster_mask.at[
            stuck_on_envs, stuck_on_thrusters
        ].set(jnp.full(self.stuck_on_envs.shape[0], PerturbationStatus.STUCK_ON.value))
        self.start_times = self.start_times.at[stuck_on_envs, stuck_on_thrusters].set(
            start_time
        )

        print(f"Thruster(s) stuck on.")
        print("Thruster mask: ", Perturbation.thruster_mask)

    def key_callback(self, keycode: Optional[int] = None) -> None:
        # Call correct method for key callbacks
        self.stuck_on_thruster()

    def replace_with_upper_thrust_bound(
        self, input_row: jnp.ndarray, start_times: jnp.ndarray, timestamp: float
    ) -> jnp.ndarray:
        """
        Return input row with updated thrust values.
        """
        # Apply the perturbation
        updated_input_row = input_row
        for i in range(len(input_row)):
            if (
                input_row[i].val[0] == PerturbationStatus.STUCK_ON.value
                and timestamp >= start_times[i]
            ):
                updated_input_row = input_row.at[i].set(
                    self.model_config.Thrusters.thruster_list[i].forcerange[1]
                )  # Max. thruster force
        return updated_input_row


"""
This section consists of "physically-grounded" perturbations modelled by a GP.
Note that these function take in the demanded force and output the actual force.
"""


class ThrusterFailureSimulator:
    """
    This class is used to the data for the nonlinear perturbations.
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
        self.demanded_force = jnp.linspace(
            0, upper_bound, num_points
        )  # Values between 0 and upper_bound
        self.failure_modes = self._get_failure_modes()

    class ThrusterFailureGPModel(gpflow.models.GPR):
        def __init__(
            self,
            train_x,
            train_y,
            kernel_type="RBF",
            lengthscale=0.2,
            outputscale=1.0,
        ):
            train_x = tf.convert_to_tensor(train_x, dtype=tf.float64)
            train_x_reshaped = tf.reshape(train_x, [-1, 1])
            train_y = tf.convert_to_tensor(train_y, dtype=tf.float64)
            train_y_reshaped = tf.reshape(train_y, [-1, 1])
            
            kernel = self._choose_kernel(kernel_type, lengthscale)
            kernel.variance.assign(outputscale)

            mean_function = gpflow.mean_functions.Constant()

            super().__init__((train_x_reshaped, train_y_reshaped), kernel=kernel, mean_function=mean_function)

        def _choose_kernel(self, kernel_type, lengthscale):
            if kernel_type == "RBF":
                return gpflow.kernels.SquaredExponential(lengthscales=lengthscale)
            elif kernel_type == "Matern":
                return gpflow.kernels.Matern32(lengthscales=lengthscale)
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        def forward(self, x):
            test_x = tf.convert_to_tensor(x, dtype=tf.float64)
            
            mean_x, covar_x = self.predict_f(tf.reshape(test_x, [-1,1]), full_cov=True)
            scale = tf.linalg.cholesky(covar_x[0, :, :])

            return tfp.distributions.MultivariateNormalTriL(
                loc=mean_x[:, 0], scale_tril=scale
            )

    def generate_failure_data(self, failure_type):
        actual_force = self._get_actual_force(failure_type)
        subset_indices = self._select_subset_indices()
        demanded_force_subset = self.demanded_force[subset_indices]
        actual_force_subset = actual_force[subset_indices]

        # Initialize GP model and likelihood
        params = self.failure_modes[failure_type]

        model = self.ThrusterFailureGPModel(
            demanded_force_subset,
            actual_force_subset,
            kernel_type=params["kernel"],
            lengthscale=params["lengthscale"],
            outputscale=params["outputscale"],
        )

        # Train model
        opt = gpflow.optimizers.Scipy()
        opt.minimize(model.training_loss, model.trainable_variables)

        # No training needed as hyperparameters are manually set
        x_test = jnp.linspace(0, self.upper_bound, self.num_points)
        sampled_function = self._sample_gp_function(
            model, x_test, failure_type
        )

        sampled_function = jax.lax.clamp(0.0, jnp.asarray(sampled_function), self.upper_bound)

        return x_test, sampled_function

    def _get_actual_force(self, failure_type):
        key = jax.random.PRNGKey(42)
        if failure_type == PerturbationStatus.SATURATED_THRUST:
            return jnp.where(
                self.demanded_force < 0.3 * self.upper_bound,
                self.demanded_force,
                0.3 * self.upper_bound
                + 0.0005
                * jax.random.normal(key, shape=self.demanded_force.shape)
                * (self.upper_bound - self.demanded_force),
            )
        elif failure_type == PerturbationStatus.FAULTY_VALVE:
            valve_min, valve_max = self.valve_min, self.valve_max
            lower_threshold = 0.67 * self.upper_bound
            upper_threshold = 0.73 * self.upper_bound
            actual_force = jnp.full_like(self.demanded_force, valve_min)
            linear_region = (self.demanded_force >= lower_threshold) & (
                self.demanded_force <= upper_threshold
            )
            actual_force = actual_force.at[linear_region].set(
                valve_min
                + (
                    (self.demanded_force[linear_region] - lower_threshold)
                    / (upper_threshold - lower_threshold)
                )
                * (valve_max - valve_min)
            )
            actual_force = actual_force.at[self.demanded_force > upper_threshold].set(valve_max)
            return actual_force
        elif failure_type == PerturbationStatus.THRUST_INSTABILITY:
            return (
                self.demanded_force
                - 0.1 * self.upper_bound * jnp.sin(10 * self.demanded_force)
                + 0.05
                * self.upper_bound
                * jax.random.normal(key, shape=self.demanded_force.shape)
            )
        elif failure_type == "thermal_stress":
            return self.demanded_force * jnp.exp(
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

    def _sample_gp_function(self, gp_model, demanded_force, failure_type):
        observed_pred = gp_model.forward(demanded_force)

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
    def __init__(
        self, env_config: BaseEnvConfig, model_config: BaseModelConfig, failure_type
    ) -> None:
        super().__init__(env_config, model_config)

        self.failure_type = failure_type
        self.start_times = jnp.zeros((self.num_envs, self.nu))
        self.thruster_list = deepcopy(model_config.Thrusters.thruster_list)

        # Store interpolation functions for each thruster
        self.interpolations = [None] * self.nu

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = None
    ) -> jnp.ndarray:
        input = input.reshape(
            self.num_envs, self.nu
        )  # Reshape to account for multiple envs

        # Do a elementwise AND operation
        faulty = jnp.logical_and(
            Perturbation.thruster_mask == self.failure_type,
            timestamp >= self.start_times,
        )

        # Apply interpolation only to the affected thrusters with active perturbations
        if jnp.any(faulty):
            # Gather input values for the affected thrusters
            affected_inputs = input[faulty]

            # Apply the respective interpolation functions
            interpolated_values = jnp.array(
                [
                    self.interpolations[i](affected_inputs[idx])
                    for idx, i in enumerate(jnp.where(faulty)[0])
                ]
            )

            # Update the input array with interpolated values
            input[faulty] = interpolated_values

        return input

    def register_perturbation(
        self,
        perturbed_envs: Optional[jnp.ndarray] = None,
        index: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        valve_min: Optional[float] = None,
        valve_max: Optional[float] = None,
    ) -> None:
        """
        Register a perturbation and store the interpolation data.
        NOTE: the same thruster fails in all chosen envs for now.
        """
        if index is not None:
            thruster_index = index
        else:
            # If we must choose the same thruster for all envs, some must be overridden
            thruster_index = np.random.randint(8)

        gp_perturbed_envs = self.get_perturbed_envs(1.0, perturbed_envs)

        Perturbation.thruster_mask = Perturbation.thruster_mask.at[
            gp_perturbed_envs, jnp.full(gp_perturbed_envs.shape[0], thruster_index)
        ].set(self.failure_type.value)
        self.start_times = self.start_times.at[
            gp_perturbed_envs, jnp.full(gp_perturbed_envs.shape[0], thruster_index)
        ].set(start_time)

        if not isinstance(valve_min, jnp.ndarray):
            valve_min = 0.15 * self.thruster_list[thruster_index].ctrlrange[-1]

        if not isinstance(valve_max, jnp.ndarray):
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
        # self.interpolations[gp_perturbed_envs, thruster_index] = interp1d(
        #     x_data, y_data, kind="linear", fill_value="extrapolate"
        # ) TODO
        self.interpolations[thruster_index] = interp1d(
            x_data, y_data, kind="linear", fill_value="extrapolate"
        )

        print(
            f"Thruster(s) are affected by a faulty valve starting at {start_time} seconds."
        )

    def key_callback(self, keycode: Optional[int] = None) -> None:
        pass


class FaultyValve(GPPerturbation):
    def __init__(
        self, env_config: BaseEnvConfig, model_config: BaseModelConfig
    ) -> None:
        super().__init__(env_config, model_config, PerturbationStatus.FAULTY_VALVE)

    def register_perturbation(
        self,
        perturbed_envs: Optional[jnp.ndarray] = None,
        index: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        valve_min: Optional[float] = None,
        valve_max: Optional[float] = None,
    ) -> None:
        if valve_min is None or valve_max is None:
            warnings.warn(
                "No min or max value set for the FaultyValve perturbation.", UserWarning
            )
        return super().register_perturbation(
            perturbed_envs, index, start_time, valve_min, valve_max
        )


class SaturatedThrust(GPPerturbation):
    def __init__(
        self, env_config: BaseEnvConfig, model_config: BaseModelConfig
    ) -> None:
        super().__init__(env_config, model_config, PerturbationStatus.SATURATED_THRUST)

    def register_perturbation(
        self,
        perturbed_envs: Optional[jnp.ndarray] = None,
        index: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        valve_min: Optional[float] = None,
        valve_max: Optional[float] = None,
    ) -> None:
        return super().register_perturbation(
            perturbed_envs, index, start_time, valve_min, valve_max
        )


class ThrustInstability(GPPerturbation):
    def __init__(
        self, env_config: BaseEnvConfig, model_config: BaseModelConfig
    ) -> None:
        super().__init__(
            env_config, model_config, PerturbationStatus.THRUST_INSTABILITY
        )

    def register_perturbation(
        self,
        perturbed_envs: Optional[jnp.ndarray] = None,
        index: Optional[int] = None,
        start_time: Optional[float] = 0.0,
        valve_min: Optional[float] = None,
        valve_max: Optional[float] = None,
    ) -> None:
        return super().register_perturbation(
            perturbed_envs, index, start_time, valve_min, valve_max
        )
