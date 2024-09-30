from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List
import numpy as np
import jax
import jax.numpy as jnp

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

        self.is_active = False

        self.perturbed_envs = perturbed_envs

    def apply(
        self, input: jnp.ndarray, timestamp: Optional[float] = 0.0
    ) -> jnp.ndarray:
        input = input.reshape(
            self.num_envs, self.nu
        )  # Reshape to account for multiple envs
        if self.is_active and timestamp >= self.start_time:
            input = input.at[
                self.thruster_mask == PerturbationStatus.STUCK_OFF.value
            ].set(0.0)

        return input

    def stuck_off_thruster(
        self, index: Optional[int] = None, start_time: Optional[float] = 0.0
    ) -> None:
        """
        Method to shut off a random thruster or a specific one if provided.
        """
        self.start_time = start_time
        self.is_active = True
        self.perturbed_envs = self.get_perturbed_envs(0.2)
        self.thruster_mask = self.thruster_mask.at[self.perturbed_envs, :].set(
            self.select_thrusters(
                self.thruster_mask[self.perturbed_envs, :],
                jnp.full(
                    self.perturbed_envs.shape[0], PerturbationStatus.STUCK_OFF.value
                ),
                index,
            )
        )
        print(f"Thruster(s) stuck off.")

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

        self.model_config = model_config

        self.failure_type = PerturbationStatus.STUCK_ON

        self.is_active = False

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
        self.start_time = start_time
        self.is_active = True
        self.perturbed_envs = self.get_perturbed_envs(0.2)
        self.thruster_mask = self.thruster_mask.at[self.perturbed_envs, :].set(
            self.select_thrusters(
                self.thruster_mask[self.perturbed_envs, :],
                jnp.full(
                    self.perturbed_envs.shape[0], PerturbationStatus.STUCK_ON.value
                ),
                index,
            )
        )
        print(f"Thruster(s) stuck on.")

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
