from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Optional, Dict, Any, Tuple
import numpy as np
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

    def __init__(self, key: jnp.ndarray) -> None:
        self._key = key
        self.state: Optional["DisturbanceState"] = None

    def _split_keys(self, count: int = 1) -> jnp.ndarray:
        """
        Split the internal RNG key and return ``count`` subkeys.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        splits = jax.random.split(self._key, count + 1)
        self._key = splits[0]
        return splits[1:]

    def _update_state(
        self,
        active_mask: jnp.ndarray,
        start_times: jnp.ndarray,
        params: Dict[str, Any],
    ) -> None:
        self.state = DisturbanceState(
            rng=self._key,
            active_mask=active_mask,
            start_times=start_times,
            params=params,
        )

    @abstractmethod
    def apply(self, timestamp: Optional[float] = 0.0) -> jnp.ndarray:
        pass

    @abstractmethod
    def key_callback(self, keycode: Optional[int] = None) -> None:
        pass


@dataclass
class DisturbanceState:
    rng: jnp.ndarray
    active_mask: jnp.ndarray
    start_times: jnp.ndarray
    params: dict


def _disturbance_state_flatten(state: "DisturbanceState"):
    children = (
        state.rng,
        state.active_mask,
        state.start_times,
        state.params,
    )
    return children, None


def _disturbance_state_unflatten(aux_data, children):
    rng, active_mask, start_times, params = children
    return DisturbanceState(
        rng=rng,
        active_mask=active_mask,
        start_times=start_times,
        params=params,
    )


jax.tree_util.register_pytree_node(
    DisturbanceState,
    _disturbance_state_flatten,
    _disturbance_state_unflatten,
)


def disturbance_state_to_serializable(
    state: Optional["DisturbanceState"],
) -> Optional[dict]:
    if state is None:
        return None
    to_np = lambda x: None if x is None else np.asarray(x)
    params_np = jax.tree_util.tree_map(to_np, state.params)
    return {
        "rng": np.asarray(state.rng),
        "active_mask": np.asarray(state.active_mask),
        "start_times": np.asarray(state.start_times),
        "params": params_np,
    }


def disturbance_state_from_serializable(
    payload: Optional[dict],
) -> Optional["DisturbanceState"]:
    if payload is None:
        return None
    to_jnp = lambda x: None if x is None else jnp.asarray(x)
    params = jax.tree_util.tree_map(to_jnp, payload["params"])
    return DisturbanceState(
        rng=jnp.asarray(payload["rng"]),
        active_mask=jnp.asarray(payload["active_mask"]),
        start_times=jnp.asarray(payload["start_times"]),
        params=params,
    )


def constant_force_apply_from_state(
    state: Optional[DisturbanceState],
    timestamp: float,
    const_force: jnp.ndarray,
) -> Tuple[jnp.ndarray, Optional[DisturbanceState]]:
    if state is None or const_force is None:
        return jnp.zeros_like(const_force), state

    active_mask = state.active_mask.astype(const_force.dtype)
    time_mask = (timestamp >= state.start_times).astype(const_force.dtype)
    force = const_force * active_mask[:, None] * time_mask[:, None]
    return force, state


def constant_force_activate_state(
    state: Optional[DisturbanceState],
    env_indices: jnp.ndarray,
    start_times: jnp.ndarray,
    const_force: jnp.ndarray,
    rng: jnp.ndarray,
) -> DisturbanceState:
    num_envs = const_force.shape[0]
    if state is None:
        active_mask = jnp.zeros((num_envs,), dtype=bool)
        current_start_times = jnp.zeros((num_envs,), dtype=start_times.dtype)
    else:
        active_mask = state.active_mask
        current_start_times = state.start_times

    if env_indices is not None and env_indices.size > 0:
        active_mask = active_mask.at[env_indices].set(True)
        current_start_times = current_start_times.at[env_indices].set(
            start_times[env_indices]
        )

    return DisturbanceState(
        rng=rng,
        active_mask=active_mask,
        start_times=current_start_times,
        params={"const_force": const_force},
    )


class DisturbanceList(ABC):
    """
    Applies multiple disturbances sequentially
    """

    def __init__(self, disturbances: list[Disturbance]) -> None:
        super().__init__()
        self.disturbances = disturbances

        # Initialize look-up dictionary for keycodes and disturbances
        self.keycode_dict = {
            "/": {
                "description": "const_force_disturbance",
                "type": 0,
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

    def key_callback(self, keycode: Optional[int] = None) -> None:
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
            if (
                disturbance.failure_type.value
                == self.keycode_dict[chr(keycode)]["type"]
            ):
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
        key: jnp.ndarray,
        magnitude: jnp.ndarray | None = None,
        direction: jnp.ndarray | None = None,
    ) -> None:
        super().__init__(key)

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
            mag_key = self._split_keys(1)[0]
            magnitude = jax.random.uniform(
                mag_key, (self.num_envs, 1), minval=0, maxval=0.2
            )

        # Randomize the force direction over the enviroments
        if direction is None:
            dir_keys = self._split_keys(3)
            x = jax.random.uniform(dir_keys[0], (self.num_envs,), minval=0, maxval=1)
            y = jax.random.uniform(dir_keys[1], (self.num_envs,), minval=0, maxval=1)
            z = jax.random.uniform(dir_keys[2], (self.num_envs,), minval=0, maxval=1)
            direction = jnp.array([x, y, z]).transpose()

        # Normalize all force directions in one vectorized operation. A Python
        # loop here is very expensive when thousands of RL envs are created.
        direction_norm = jnp.linalg.norm(direction, axis=1, keepdims=True)
        direction = direction / jnp.maximum(direction_norm, 1e-8)

        self.magnitude = magnitude
        self.direction = direction

        # Save constant disturbance as 6D array
        force, torque = self.magnitude * self.direction, jnp.zeros((self.num_envs, 3))
        self.const_force = jnp.concatenate((force, torque), axis=1)
        self._update_state(
            active_mask=jnp.zeros((self.num_envs,), dtype=bool),
            start_times=self.start_times,
            params={
                "const_force": self.const_force,
            },
        )

    def apply(self, timestamp: Optional[float] = 0.0) -> jnp.ndarray:
        force, new_state = constant_force_apply_from_state(
            self.state,
            float(timestamp),
            self.const_force,
        )
        if new_state is not None:
            self.state = new_state
        return force

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
            num_disturbed = max(1, int(fraction_disturbed_envs * self.num_envs))
            perm_key = self._split_keys(1)[0]
            permutation = jax.random.permutation(perm_key, self.num_envs)
            self.disturbed_envs = permutation[:num_disturbed]

        if self.disturbed_envs is not None and self.disturbed_envs.size > 0:
            self.start_times = self.start_times.at[self.disturbed_envs].set(start_time)
        self.state = constant_force_activate_state(
            self.state,
            (
                self.disturbed_envs
                if self.disturbed_envs is not None
                else jnp.array([], dtype=int)
            ),
            self.start_times,
            self.const_force,
            self._key,
        )
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

    def to_state(self, env_config) -> DisturbanceState:
        """
        Create an immutable state snapshot for functional helpers.
        """
        num_envs = (
            env_config.control.RL.num_envs if hasattr(env_config.control, "RL") else 1
        )
        return DisturbanceState(
            rng=self._key,
            active_mask=jnp.zeros((num_envs,), dtype=bool),
            start_times=jnp.zeros((num_envs,)),
            params={},
        )
