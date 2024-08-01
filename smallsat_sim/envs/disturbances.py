from abc import ABC, abstractmethod
import jax
import jax.numpy as jnp


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

    def apply(self, current_sim_time):
        # Net force vector of all the disturbances in the list
        self.net_force = jnp.zeros(6)

        for disturbance in self.disturbances:
            self.net_force += disturbance.apply(current_sim_time)

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
        print(
            self.keycode_dict.get(chr(keycode), {}).get(
                "warning", "No disturbance found for keycode."
            )
        )
        return None

    def _get_disturbance_index_from_string(
        self, desired_disturbance: str
    ) -> None | int:
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
        self, env_config, magnitude=None, direction=None, disturbed_envs=None
    ) -> None:
        super().__init__()

        self.failure_type = "const_force_disturbance"

        self.is_active = False

        # Find out how many environments there are
        if hasattr(env_config.control, "RL"):
            self.num_envs = env_config.control.RL.num_envs
        else:
            self.num_envs = 1

        # Randomize the force magnitude over the enviroments
        if magnitude is None:
            magnitude = jax.random.uniform(
                jax.random.PRNGKey(42), (self.num_envs, 1), minval=0, maxval=0.2
            )

        if disturbed_envs is None:
            fraction_disturbed_envs = 0.1
            self.disturbed_envs = jax.random.randint(
                jax.random.PRNGKey(1),
                shape=(int(fraction_disturbed_envs * self.num_envs),),
                minval=0,
                maxval=self.num_envs,
            )
        else:
            self.disturbed_envs = disturbed_envs

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

    def apply(self, current_sim_time) -> jnp.ndarray:
        if self.is_active and current_sim_time >= self.start_time:
            mask = jnp.zeros_like(self.const_force)
            mask = mask.at[self.disturbed_envs, :].set(1)
            return self.const_force * mask
        else:
            return jnp.zeros((self.num_envs, 6))

    def const_force_disturbance(self, start_time=0.0) -> None:
        """
        Activate the constant force disturbance.
        """
        self.start_time = start_time
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
