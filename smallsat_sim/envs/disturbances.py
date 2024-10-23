from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional, List
import jax
import jax.numpy as jnp


class DisturbanceStatus(Enum):
    """
    Document type of active disturbance.
    0 := Constant force disturbance
    """

    CONSTANT_FORCE = 0


class Disturbance(ABC):
    """
    Base class for all disturbances applied to the model.
    Disturbances are defined as an external influence,
    which apply a force or torque on the model.
    """

    def __init__(self) -> None:
        pass

    @abstractmethod
    def apply(self) -> jnp.ndarray:
        pass

    @abstractmethod
    def key_callback(self, keycode: Optional[int] = None) -> None:
        pass


class DisturbanceList(ABC):
    """
    Applies multiple disturbances sequentially
    """

    def __init__(self, disturbances: List[Disturbance]) -> None:
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

    def apply(self, timestamp: Optional[float] = 0.0):
        # Net force vector of all the disturbances in the list
        self.net_force = jnp.zeros(6)

        for disturbance in self.disturbances:
            self.net_force += disturbance.apply(timestamp)

        return self.net_force

    def key_callback(self, keycode: Optional[int] = None):
        """
        Handles external keycall back calls based on registered disturbance modules inside of self.disturbances.
        """
        # Extract location of desired disturbance (if it exists)
        disturbance_index = self._check_registered_disturbances(keycode)

        # Apply the callback function
        if isinstance(disturbance_index, int):
            self.disturbances[disturbance_index].key_callback(keycode)

    def _check_registered_disturbances(
        self, keycode: Optional[int] = None
    ) -> Optional[int]:
        # Iterate over disturbances to find a matching type
        for idx, disturbance in enumerate(self.disturbances):
            if disturbance.failure_type == self.keycode_dict[chr(keycode)]["type"]:
                return idx

        # Print warning if no matching disturbance is found and return None
        print(
            self.keycode_dict.get(chr(keycode), {}).get(
                "warning", "No disturbance found for keycode."
            )
        )
        return None

    def _get_disturbance_index_from_string(
        self, desired_disturbance: str
    ) -> Optional[int]:
        # Iterate over disturbances to find a matching type
        for idx, disturbance in enumerate(self.disturbances):
            if disturbance.failure_type == desired_disturbance:
                return idx

        # Print warning if no matching perturbation is found and return None
        print("warning", "No perturbation found for keycode.")
        return None


class ConstantForceDisturbance(Disturbance):
    """
    Applies a constant force disturbance in a specified direction. If an argument is 'None', its value is randomly chosen.
    """

    def __init__(
        self,
        env_config,
        magnitude: Optional[float] = None,
        direction: Optional[float] = None,
    ) -> None:
        super().__init__()

        self.failure_type = DisturbanceStatus.CONSTANT_FORCE
        self.is_active = False

        # Find out how many environments there are
        if hasattr(env_config.control, "RL"):
            self.num_envs = env_config.control.RL.num_envs
        else:
            self.num_envs = 1

        self.disturbed_envs = None

        self.start_times = jnp.zeros(self.num_envs)

        # Randomize the force magnitude over the enviroments
        if magnitude is None:
            magnitude = jax.random.uniform(
                jax.random.PRNGKey(42), (self.num_envs, 1), minval=0, maxval=0.2
            )

        # Randomize the force direction over the enviroments
        if direction is None:
            x = jax.random.uniform(
                jax.random.PRNGKey(0), (self.num_envs,), minval=0, maxval=1
            )
            y = jax.random.uniform(
                jax.random.PRNGKey(1), (self.num_envs,), minval=0, maxval=1
            )
            z = jax.random.uniform(
                jax.random.PRNGKey(2), (self.num_envs,), minval=0, maxval=1
            )
            direction = jnp.array([x, y, z]).transpose()

        # Normalize the force direction
        for i in range(direction.shape[0]):
            direction = direction.at[i, :].set(
                direction[i, :] / jnp.linalg.norm(direction[i, :])
            )

        self.magnitude = magnitude
        self.direction = direction

        # Save constant disturbance as 6D array
        force, torque = self.magnitude * self.direction, jnp.zeros((self.num_envs, 3))
        self.const_force = jnp.concatenate((force, torque), axis=1)

    def apply(self, timestamp: Optional[float] = 0.0) -> jnp.ndarray:
        disturbance_mask, time_mask = jnp.zeros_like(self.const_force), jnp.zeros_like(
            self.const_force
        )
        if self.disturbed_envs is not None:
            disturbance_mask = disturbance_mask.at[self.disturbed_envs, :].set(1)
        time_mask = time_mask.at[timestamp >= self.start_times, :].set(1)
        return self.const_force * disturbance_mask * time_mask

    def const_force_disturbance(
        self,
        disturbed_envs: Optional[jnp.ndarray] = None,
        start_time: Optional[float] = 0.0,
    ) -> None:
        """
        Activate the constant force disturbance.
        """
        if isinstance(disturbed_envs, jnp.ndarray):
            self.disturbed_envs = disturbed_envs
        else:
            fraction_disturbed_envs = 1.0
            self.disturbed_envs = jax.random.randint(
                jax.random.PRNGKey(42),
                shape=(int(fraction_disturbed_envs * self.num_envs),),
                minval=0,
                maxval=self.num_envs,
            )

        self.start_times = self.start_times.at[disturbed_envs].set(start_time)
        print("Constant force disturbance is active.")

    def deactivate_const_force_disturbance(self) -> None:
        """
        Reset disturbance.
        """
        self.is_active = False
        print("Constant force disturbance is inactive.")

    def key_callback(self, keycode: Optional[int] = None) -> None:
        # Call correct method for key callbacks
        self.const_force_disturbance()
