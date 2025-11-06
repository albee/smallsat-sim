from abc import ABC, abstractmethod
from dataclasses import dataclass, replace
from enum import Enum
import warnings
from copy import deepcopy
from typing import Optional, Tuple
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import numpy as np

from smallsat_sim.envs.base_env_config import BaseEnvConfig
from smallsat_sim.model.base_model_config import BaseModelConfig

# Import for typing
from smallsat_sim.model.base_model_config import BaseModelConfig


@dataclass
class PerturbationState:
    rng: jnp.ndarray
    thruster_mask: jnp.ndarray
    failure_value: Optional[int] = None
    start_times: Optional[jnp.ndarray] = None
    max_thruster_force: Optional[jnp.ndarray] = None
    gp_x_samples: Optional[jnp.ndarray] = None
    gp_y_samples: Optional[jnp.ndarray] = None


def _perturbation_state_flatten(state: "PerturbationState"):
    children = (
        state.rng,
        state.thruster_mask,
        state.start_times,
        state.max_thruster_force,
        state.gp_x_samples,
        state.gp_y_samples,
    )
    return children, state.failure_value


def _perturbation_state_unflatten(aux_data, children):
    (
        rng,
        thruster_mask,
        start_times,
        max_thruster_force,
        gp_x_samples,
        gp_y_samples,
    ) = children
    return PerturbationState(
        rng=rng,
        thruster_mask=thruster_mask,
        failure_value=aux_data,
        start_times=start_times,
        max_thruster_force=max_thruster_force,
        gp_x_samples=gp_x_samples,
        gp_y_samples=gp_y_samples,
    )


jax.tree_util.register_pytree_node(
    PerturbationState,
    _perturbation_state_flatten,
    _perturbation_state_unflatten,
)


def perturbation_state_to_serializable(
    state: Optional["PerturbationState"],
) -> Optional[dict]:
    if state is None:
        return None

    to_np = lambda x: None if x is None else np.asarray(x)

    return {
        "rng": np.asarray(state.rng),
        "thruster_mask": np.asarray(state.thruster_mask),
        "failure_value": (
            None if state.failure_value is None else int(state.failure_value)
        ),
        "start_times": to_np(state.start_times),
        "max_thruster_force": to_np(state.max_thruster_force),
        "gp_x_samples": to_np(state.gp_x_samples),
        "gp_y_samples": to_np(state.gp_y_samples),
    }


def perturbation_state_from_serializable(
    payload: Optional[dict],
) -> Optional["PerturbationState"]:
    if payload is None:
        return None

    to_jnp = lambda x: None if x is None else jnp.asarray(x)

    return PerturbationState(
        rng=jnp.asarray(payload["rng"]),
        thruster_mask=jnp.asarray(payload["thruster_mask"]),
        failure_value=payload["failure_value"],
        start_times=to_jnp(payload["start_times"]),
        max_thruster_force=to_jnp(payload["max_thruster_force"]),
        gp_x_samples=to_jnp(payload["gp_x_samples"]),
        gp_y_samples=to_jnp(payload["gp_y_samples"]),
    )


def _broadcast_to_control_shape(arr: jnp.ndarray, control: jnp.ndarray) -> jnp.ndarray:
    """
    Utility to broadcast stored perturbation arrays (which may be recorded per-env or per-thruster)
    to the full control array shape.
    """
    arr = jnp.asarray(arr)
    control = jnp.asarray(control)
    control_ndim = control.ndim
    if control_ndim == 0:
        return jnp.asarray(arr)
    if arr.shape == control.shape:
        return arr

    if arr.ndim == 0:
        arr = jnp.reshape(arr, (1,) * control_ndim)
    elif arr.ndim == 1:
        if arr.shape[0] == control.shape[-1]:
            arr = jnp.reshape(arr, (1, control.shape[-1]))
        elif arr.shape[0] == control.shape[0]:
            arr = jnp.reshape(arr, (control.shape[0], 1))
        else:
            arr = jnp.reshape(arr, (1,) * (control_ndim - 1) + arr.shape)
    elif arr.ndim == control_ndim - 1:
        if arr.shape[-1] == control.shape[-1]:
            arr = jnp.reshape(arr, (1,) + arr.shape)
        elif arr.shape[0] == control.shape[0]:
            arr = jnp.reshape(arr, arr.shape + (1,))
        else:
            arr = jnp.reshape(arr, (1,) * (control_ndim - arr.ndim) + arr.shape)
    elif arr.ndim > control_ndim:
        arr = jnp.reshape(arr, arr.shape[-control_ndim:])
    else:
        arr = jnp.reshape(arr, (1,) * (control_ndim - arr.ndim) + arr.shape)

    return jnp.broadcast_to(arr, control.shape)


def stuck_off_apply_from_state(
    state: Optional[PerturbationState],
    control: jnp.ndarray,
    timestamp: float,
) -> Tuple[jnp.ndarray, Optional[PerturbationState]]:
    if state is None:
        return control, state
    control = jnp.asarray(control)
    mask = jnp.asarray(state.thruster_mask)
    if mask.ndim == 1:
        if mask.shape[0] == control.shape[0]:
            mask = mask[:, None]
        elif mask.shape[0] == control.shape[-1]:
            mask = mask[None, :]
        else:
            mask = mask.reshape((1,) * (control.ndim - mask.ndim) + mask.shape)
    elif mask.ndim < control.ndim:
        mask = mask.reshape((1,) * (control.ndim - mask.ndim) + mask.shape)
    mask = jnp.broadcast_to(mask, control.shape)
    mask = mask == PerturbationStatus.STUCK_OFF.value

    if state.start_times is not None:
        start_times = _broadcast_to_control_shape(state.start_times, control)
    else:
        start_times = jnp.zeros(control.shape, dtype=control.dtype)

    timestamp_arr = _broadcast_to_control_shape(
        jnp.asarray(timestamp, dtype=start_times.dtype),
        control,
    )
    active = jnp.logical_and(mask, timestamp_arr >= start_times)
    control = jnp.where(active, 0.0, control)
    return control, state


def stuck_off_activate_state(
    state: Optional[PerturbationState],
    thruster_mask: jnp.ndarray,
    start_times: jnp.ndarray,
    rng: jnp.ndarray,
) -> PerturbationState:
    return PerturbationState(
        rng=rng,
        thruster_mask=thruster_mask,
        failure_value=PerturbationStatus.STUCK_OFF.value,
        start_times=start_times,
        max_thruster_force=state.max_thruster_force if state else None,
        gp_x_samples=state.gp_x_samples if state else None,
        gp_y_samples=state.gp_y_samples if state else None,
    )


def stuck_on_apply_from_state(
    state: Optional[PerturbationState],
    control: jnp.ndarray,
    timestamp: float,
) -> Tuple[jnp.ndarray, Optional[PerturbationState]]:
    if state is None:
        return control, state

    start_times = state.start_times
    max_force = state.max_thruster_force
    if start_times is None or max_force is None:
        return control, state

    control = jnp.asarray(control)
    mask = _broadcast_to_control_shape(state.thruster_mask, control)
    mask = mask == PerturbationStatus.STUCK_ON.value

    start_times = _broadcast_to_control_shape(start_times, control)

    timestamp_arr = _broadcast_to_control_shape(
        jnp.asarray(timestamp, dtype=start_times.dtype),
        control,
    )
    active = jnp.logical_and(mask, timestamp_arr >= start_times)
    broadcast_force = _broadcast_to_control_shape(max_force, control)
    control = jnp.where(active, broadcast_force, control)
    return control, state


def stuck_on_activate_state(
    state: Optional[PerturbationState],
    thruster_mask: jnp.ndarray,
    start_times: jnp.ndarray,
    max_thruster_force: jnp.ndarray,
    rng: jnp.ndarray,
) -> PerturbationState:
    return PerturbationState(
        rng=rng,
        thruster_mask=thruster_mask,
        failure_value=PerturbationStatus.STUCK_ON.value,
        start_times=start_times,
        max_thruster_force=max_thruster_force,
        gp_x_samples=state.gp_x_samples if state else None,
        gp_y_samples=state.gp_y_samples if state else None,
    )


def gp_apply_from_state(
    state: Optional[PerturbationState],
    control: jnp.ndarray,
    timestamp: float,
    failure_value: int,
) -> Tuple[jnp.ndarray, Optional[PerturbationState]]:
    if (
        state is None
        or state.start_times is None
        or state.gp_x_samples is None
        or state.gp_y_samples is None
    ):
        return control, state

    control = jnp.asarray(control)
    mask = _broadcast_to_control_shape(state.thruster_mask, control)
    mask = mask == failure_value

    start_times = _broadcast_to_control_shape(state.start_times, control)

    timestamp_arr = _broadcast_to_control_shape(
        jnp.asarray(timestamp, dtype=start_times.dtype),
        control,
    )
    active = jnp.logical_and(mask, timestamp_arr >= start_times)

    def _interp_single(args):
        active_flag, value, xs, ys = args
        return jax.lax.cond(
            active_flag,
            lambda tup: jnp.interp(tup[0], tup[1], tup[2]),
            lambda tup: tup[0],
            (value, xs, ys),
        )

    def _interp_row(ctrl_row, active_row):
        return jax.vmap(
            lambda a, v, xs, ys: _interp_single((a, v, xs, ys)),
            in_axes=(0, 0, 0, 0),
        )(active_row, ctrl_row, state.gp_x_samples, state.gp_y_samples)

    control = jax.vmap(
        lambda ctrl_row, active_row: _interp_row(ctrl_row, active_row),
        in_axes=(0, 0),
    )(control, active)

    return control, state


def gp_register_state(
    state: Optional[PerturbationState],
    thruster_mask: jnp.ndarray,
    start_times: jnp.ndarray,
    rng: jnp.ndarray,
    failure_value: int,
    thruster_index: int,
    x_samples: jnp.ndarray,
    y_samples: jnp.ndarray,
) -> PerturbationState:
    num_thrusters = thruster_mask.shape[1]
    num_points = x_samples.shape[0]

    if state is not None and state.gp_x_samples is not None:
        gp_x = state.gp_x_samples
        gp_y = state.gp_y_samples
    else:
        gp_x = jnp.zeros((num_thrusters, num_points), dtype=x_samples.dtype)
        gp_y = jnp.zeros_like(gp_x)

    gp_x = gp_x.at[thruster_index].set(x_samples)
    gp_y = gp_y.at[thruster_index].set(y_samples)

    return PerturbationState(
        rng=rng,
        thruster_mask=thruster_mask,
        failure_value=failure_value,
        start_times=start_times,
        max_thruster_force=state.max_thruster_force if state else None,
        gp_x_samples=gp_x,
        gp_y_samples=gp_y,
    )


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
        self,
        env_config: BaseEnvConfig,
        model_config: BaseModelConfig,
        key: jnp.ndarray,
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

        self._key = key
        self.state: Optional[PerturbationState] = PerturbationState(
            rng=self._key,
            thruster_mask=Perturbation.thruster_mask,
        )

    def _split_keys(self, key: jnp.ndarray | None, count: int) -> jnp.ndarray:
        """
        Split either the provided key or the internal PRNG.
        Returns ``count`` subkeys and updates internal state to the residual key.
        """
        if count < 1:
            raise ValueError("count must be >= 1")
        source = self._key if key is None else key
        splits = jax.random.split(source, count + 1)
        self._key = splits[0]
        if self.state is not None:
            self.state = replace(self.state, rng=self._key)
        return splits[1:]

    def get_perturbed_envs(
        self,
        key,
        fraction_perturbed_envs: float,
        perturbed_envs: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Return array with envs where a failure occurs. If fraction_perturbed_envs and num_envs are too low, no envs will be perturbed.
        """
        if perturbed_envs is not None:
            return jnp.asarray(perturbed_envs, dtype=jnp.int32)

        fraction = float(
            jax.device_get(jnp.clip(jnp.asarray(fraction_perturbed_envs), 0.0, 1.0))
        )
        num_perturbed_envs = int(fraction * self.num_envs)

        if num_perturbed_envs <= 0:
            return jnp.array([], dtype=jnp.int32)

        operational_thruster_mask = jnp.any(
            Perturbation.thruster_mask == PerturbationStatus.OPERATIONAL.value,
            axis=1,
        )  # Only select envs that still have functioning thrusters
        operational_envs = jnp.where(operational_thruster_mask)[0]

        if operational_envs.size == 0:
            return jnp.array([], dtype=jnp.int32)

        num_perturbed_envs = min(num_perturbed_envs, int(operational_envs.size))
        perm_key = self._split_keys(key, 1)[0]
        shuffled_indices = jax.random.permutation(perm_key, operational_envs)
        return shuffled_indices[:num_perturbed_envs]

    def select_thrusters(
        self,
        key,
        perturbed_envs: jnp.ndarray,
        perturbed_thrusters: jnp.ndarray | None = None,
    ) -> jnp.ndarray:
        """
        Return thruster indices to fail.
        """

        if perturbed_thrusters is not None:
            return jnp.asarray(perturbed_thrusters, dtype=jnp.int32)

        if perturbed_envs.size == 0:
            return jnp.array([], dtype=jnp.int32)

        def random_operational_thruster(rng_key, row, max_size=self.nu):
            # Get indices of operational thrusters (pad with -1 if not enough)
            operational_thrusters = jnp.where(
                row == PerturbationStatus.OPERATIONAL.value,
                size=max_size,
                fill_value=-1,
            )[0]

            # Replace invalid (-1) indices with a large number so they won't be selected
            valid_mask = operational_thrusters != -1
            masked_indices = jnp.where(valid_mask, operational_thrusters, max_size)
            num_valid = valid_mask.sum()

            def choose_valid(_):
                probs = valid_mask.astype(jnp.float32) / num_valid
                return jax.random.choice(rng_key, masked_indices, p=probs)

            def no_valid(_):
                return jnp.int32(-1)

            return jax.lax.cond(num_valid > 0, choose_valid, no_valid, operand=None)

        selected_envs = Perturbation.thruster_mask[perturbed_envs]
        subkeys = self._split_keys(key, selected_envs.shape[0])

        vec_random_operational_thruster = jax.vmap(
            random_operational_thruster, in_axes=(0, 0)
        )
        selected_thrusters = vec_random_operational_thruster(subkeys, selected_envs)

        return selected_thrusters

    @abstractmethod
    def apply(self, input: jnp.ndarray, timestamp: float = 0.0) -> jnp.ndarray:
        pass

    @abstractmethod
    def key_callback(self, keycode: int | None = None) -> None:
        pass


class PerturbationList(ABC):
    """
    Applies multiple perturbations.
    """

    def __init__(self, perturbations: list[Perturbation]) -> None:
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
            ";": {
                "description": "sample_perturbation",
                "type": 3,
                "warning": "Could not fail thruster. "
                "No SamplePerturbation Perturbation module defined.",
            },
        }

    def apply(self, input: jnp.ndarray, timestamp: float = 0.0) -> jnp.ndarray:
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
        perturbation = self.perturbations[perturbation_idx]
        if perturbation.state is not None:
            start_times = getattr(perturbation, "start_times", None)
            perturbation.state = replace(
                perturbation.state,
                thruster_mask=perturbation.thruster_mask,
                start_times=start_times,
            )

    def key_callback(self, keycode: int | None = None):
        """
        Handles external keycall back calls based on registered perturbation modules inside of self.perturbations.
        """
        # Extract location of desired perturbation (if it exists)
        perturbation_idx = self._check_registered_perturbations(keycode)

        # Apply the callback function
        if isinstance(perturbation_idx, int):
            self.perturbations[perturbation_idx].key_callback(keycode)

    def _check_registered_perturbations(self, keycode: int | None = None) -> int | None:
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
    ) -> int | None:
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
        key: jnp.ndarray,
    ) -> None:
        super().__init__(env_config, model_config, key)

        self.failure_type = PerturbationStatus.STUCK_OFF
        self.start_times = jnp.zeros((self.num_envs, self.nu))
        self.state = PerturbationState(
            rng=self._key,
            thruster_mask=Perturbation.thruster_mask,
            failure_value=self.failure_type.value,
            start_times=self.start_times,
        )

    def apply(self, input: jnp.ndarray, timestamp: float = 0.0) -> jnp.ndarray:
        if input.shape[-1] != self.nu:
            warnings.warn(
                "StuckOffThrusters: skipping perturbation because actuator dimension does not match configuration."
            )
            return input
        input = input.reshape(
            self.num_envs, self.nu
        )  # Reshape to account for multiple envs
        adjusted, state = stuck_off_apply_from_state(self.state, input, timestamp)
        if state is not None:
            self.state = state
        return adjusted

    def stuck_off_thruster(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None = None,
        perturbed_thrusters: jnp.ndarray | None = None,
        start_time: float | None = None,
    ) -> None:
        """
        Method to shut off a random thruster or a specific one if provided.
        """
        env_key, thruster_key = jax.random.split(key)
        stuck_off_envs = self.get_perturbed_envs(env_key, 1.0, perturbed_envs)
        stuck_off_thrusters = self.select_thrusters(
            thruster_key, stuck_off_envs, perturbed_thrusters
        )

        Perturbation.thruster_mask = Perturbation.thruster_mask.at[
            stuck_off_envs, stuck_off_thrusters
        ].set(jnp.full(stuck_off_envs.shape[0], PerturbationStatus.STUCK_OFF.value))

        start_time_value = jnp.asarray(
            0.0 if start_time is None else start_time, dtype=self.start_times.dtype
        )
        self.start_times = self.start_times.at[stuck_off_envs, stuck_off_thrusters].set(
            start_time_value
        )
        self.state = stuck_off_activate_state(
            self.state,
            Perturbation.thruster_mask,
            self.start_times,
            self._key,
        )

        print(f"Thruster(s) stuck off.")
        # print("Thruster mask: ", Perturbation.thruster_mask)

    def key_callback(self, keycode: int | None = None) -> None:
        # Call correct method for key callbacks
        callback_key = self._split_keys(None, 1)[0]
        self.stuck_off_thruster(callback_key)


class StuckOnThrusters(Perturbation):
    """
    Thrusters unable to be turned off.
    """

    def __init__(
        self,
        env_config: BaseEnvConfig,
        model_config: BaseModelConfig,
        key: jnp.ndarray,
    ) -> None:
        super().__init__(env_config, model_config, key)

        self.model_config = model_config

        self.failure_type = PerturbationStatus.STUCK_ON
        self.start_times = jnp.zeros((self.num_envs, self.nu))

        self.max_thruster_force = jnp.asarray(
            [
                thruster.forcerange[1]
                for thruster in self.model_config.Thrusters.thruster_list
            ],
            dtype=self.start_times.dtype,
        )

        # Vectorized input replacement across environments
        self.replace_thrust_inputs = jax.vmap(
            self.replace_with_upper_thrust_bound, in_axes=(0, 0, 0, None)
        )
        self.state = PerturbationState(
            rng=self._key,
            thruster_mask=Perturbation.thruster_mask,
            failure_value=self.failure_type.value,
            start_times=self.start_times,
            max_thruster_force=self.max_thruster_force,
        )

    def apply(self, input: jnp.ndarray, timestamp: float = 0.0) -> jnp.ndarray:
        if input.shape[-1] != self.nu:
            warnings.warn(
                "StuckOnThrusters: skipping perturbation because actuator dimension does not match configuration."
            )
            return input
        input = input.reshape(
            self.num_envs, self.nu
        )  # Reshape to account for multiple envs

        adjusted, state = stuck_on_apply_from_state(self.state, input, timestamp)
        if state is not None:
            self.state = state
        return adjusted

    def stuck_on_thruster(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None = None,
        perturbed_thrusters: jnp.ndarray | None = None,
        start_time: float | None = None,
    ) -> None:
        """
        Method to unable a random thruster or a specific one if provided, to shut off.
        """
        env_key, thruster_key = jax.random.split(key)
        stuck_on_envs = self.get_perturbed_envs(env_key, 1.0, perturbed_envs)

        stuck_on_thrusters = self.select_thrusters(
            thruster_key, stuck_on_envs, perturbed_thrusters
        )

        Perturbation.thruster_mask = Perturbation.thruster_mask.at[
            stuck_on_envs, stuck_on_thrusters
        ].set(jnp.full(stuck_on_envs.shape[0], PerturbationStatus.STUCK_ON.value))
        start_time_value = jnp.asarray(
            0.0 if start_time is None else start_time, dtype=self.start_times.dtype
        )
        self.start_times = self.start_times.at[stuck_on_envs, stuck_on_thrusters].set(
            start_time_value
        )
        self.state = stuck_on_activate_state(
            self.state,
            Perturbation.thruster_mask,
            self.start_times,
            self.max_thruster_force,
            self._key,
        )

        print(f"Thruster(s) stuck on.")
        # print("Thruster mask: ", Perturbation.thruster_mask)

    def key_callback(self, keycode: int | None = None) -> None:
        callback_key = self._split_keys(None, 1)[0]
        self.stuck_on_thruster(callback_key)

    def replace_with_upper_thrust_bound(
        self,
        input_row: jnp.ndarray,
        mask_row: jnp.ndarray,
        start_times: jnp.ndarray,
        timestamp: float = 0.0,
    ) -> jnp.ndarray:
        """
        Return input row with updated thrust values.
        """
        timestamp_value = jnp.asarray(timestamp, dtype=start_times.dtype)
        active_mask = jnp.logical_and(
            mask_row == PerturbationStatus.STUCK_ON.value,
            timestamp_value >= start_times,
        )
        return jnp.where(active_mask, self.max_thruster_force, input_row)


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

    @dataclass
    class _GPPosterior:
        """Lightweight Gaussian distribution wrapper compatible with the old API."""

        loc: jnp.ndarray
        scale_tril: jnp.ndarray

        def mean(self) -> jnp.ndarray:
            return self.loc

        def sample(self, key) -> jnp.ndarray:
            if key is None:
                raise ValueError(
                    "A PRNGKey must be provided when sampling from the GP posterior."
                )
            normal = jax.random.normal(key, shape=self.loc.shape, dtype=self.loc.dtype)
            return self.loc + self.scale_tril @ normal

    class ThrusterFailureGPModel:
        def __init__(
            self,
            train_x,
            train_y,
            kernel_type="RBF",
            lengthscale=0.2,
            outputscale=1.0,
        ):
            dtype = jnp.float32
            train_x = jnp.asarray(train_x, dtype=dtype).reshape(-1, 1)
            train_y = jnp.asarray(train_y, dtype=dtype).reshape(-1, 1)

            self._train_x = train_x
            self._kernel = self._choose_kernel(kernel_type, lengthscale, outputscale)
            self._mean_value = float(jnp.mean(train_y))
            self._jitter = jnp.asarray(1e-6, dtype=dtype)

            centered_y = train_y - self._mean_value
            k_xx = self._kernel(train_x, train_x)
            noise = 1e-6 * jnp.eye(train_x.shape[0], dtype=k_xx.dtype)
            self._train_chol = jsp.linalg.cholesky(
                k_xx + noise + self._jitter * jnp.eye(k_xx.shape[0], dtype=k_xx.dtype),
                lower=True,
            )
            self._alpha = jsp.linalg.cho_solve((self._train_chol, True), centered_y)

        def _choose_kernel(self, kernel_type, lengthscale, outputscale):
            dtype = jnp.float32
            lengthscale = jnp.asarray(lengthscale, dtype=dtype)
            variance = jnp.asarray(outputscale, dtype=dtype)

            def _rbf(x, y):
                x = jnp.asarray(x, dtype=dtype)
                y = jnp.asarray(y, dtype=dtype)
                sq_dist = jnp.sum((x[:, None, :] - y[None, :, :]) ** 2, axis=-1)
                return variance * jnp.exp(-0.5 * sq_dist / (lengthscale**2 + 1e-12))

            def _matern32(x, y):
                x = jnp.asarray(x, dtype=dtype)
                y = jnp.asarray(y, dtype=dtype)
                scaled = jnp.sqrt(
                    jnp.sum(
                        ((x[:, None, :] - y[None, :, :]) / (lengthscale + 1e-12)) ** 2,
                        axis=-1,
                    )
                )
                sqrt3 = jnp.sqrt(jnp.asarray(3.0, dtype=dtype))
                return variance * (1.0 + sqrt3 * scaled) * jnp.exp(-sqrt3 * scaled)

            if kernel_type == "RBF":
                return _rbf
            elif kernel_type == "Matern":
                return _matern32
            raise ValueError(f"Unsupported kernel type: {kernel_type}")

        def forward(self, x):
            test_x = jnp.asarray(x, dtype=jnp.float32).reshape(-1, 1)

            k_xt = self._kernel(self._train_x, test_x)
            predictive_mean = self._mean_value + jnp.matmul(
                k_xt.T, self._alpha
            ).reshape(-1)

            v = jsp.linalg.solve_triangular(self._train_chol, k_xt, lower=True)
            k_tt = self._kernel(test_x, test_x)
            predictive_cov = k_tt - jnp.matmul(v.T, v)
            predictive_cov = (predictive_cov + predictive_cov.T) * 0.5
            predictive_cov = predictive_cov + self._jitter * jnp.eye(
                predictive_cov.shape[0], dtype=predictive_cov.dtype
            )
            scale_tril = jsp.linalg.cholesky(predictive_cov, lower=True)

            return ThrusterFailureSimulator._GPPosterior(
                loc=predictive_mean, scale_tril=scale_tril
            )

    def generate_failure_data(self, key, failure_type):
        key_actual, key_subset, key_sample = jax.random.split(key, 3)

        actual_force = self._get_actual_force(key_actual, failure_type)
        subset_indices = self._select_subset_indices(key_subset)
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

        x_test = jnp.linspace(0, self.upper_bound, self.num_points)
        sampled_function = self._sample_gp_function(
            model, x_test, failure_type, key_sample
        )

        sampled_function = jax.lax.clamp(
            0.0, jnp.asarray(sampled_function), self.upper_bound
        )

        return x_test, sampled_function

    def _get_actual_force(self, key, failure_type):
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
            actual_force = actual_force.at[self.demanded_force > upper_threshold].set(
                valve_max
            )
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

    def _select_subset_indices(self, key):
        indices = jax.random.choice(
            key,
            jnp.arange(1, self.num_points - 1),
            shape=(self.subset_size - 2,),
            replace=False,
        )
        start = jnp.array([0], dtype=indices.dtype)
        end = jnp.array([self.num_points - 1], dtype=indices.dtype)
        indices = jnp.concatenate([start, indices, end])
        return jnp.sort(indices)

    def _sample_gp_function(self, gp_model, demanded_force, failure_type, sample_key):
        observed_pred = gp_model.forward(demanded_force)

        if failure_type == PerturbationStatus.SATURATED_THRUST:
            return observed_pred.mean()
        else:
            return observed_pred.sample(sample_key)

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
        self,
        env_config: BaseEnvConfig,
        model_config: BaseModelConfig,
        key: jnp.ndarray,
        failure_type,
    ) -> None:
        super().__init__(env_config, model_config, key)

        self.failure_type = failure_type
        self.start_times = jnp.zeros((self.num_envs, self.nu))
        self.thruster_list = deepcopy(model_config.Thrusters.thruster_list)
        self.state = PerturbationState(
            rng=self._key,
            thruster_mask=Perturbation.thruster_mask,
            failure_value=self.failure_type.value,
            start_times=self.start_times,
        )

    def apply(self, input: jnp.ndarray, timestamp: float = 0.0) -> jnp.ndarray:
        if input.shape[-1] != self.nu:
            warnings.warn(
                f"{self.__class__.__name__}: skipping perturbation because actuator dimension does not match configuration."
            )
            return input
        reshaped = input.reshape(self.num_envs, self.nu)
        adjusted, new_state = gp_apply_from_state(
            self.state,
            reshaped,
            timestamp,
            self.failure_type.value,
        )
        if new_state is not None:
            self.state = new_state
        return adjusted

    def register_perturbation(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None = None,
        index: int | None = None,
        start_time: float | None = None,
        valve_min: float | None = None,
        valve_max: float | None = None,
    ) -> None:
        """
        Register a perturbation and store the interpolation data.
        NOTE: the same thruster fails in all chosen envs for now.
        """
        if index is not None:
            thruster_index = index
            env_key = key
        else:
            thruster_key, env_key = jax.random.split(key)
            thruster_index = int(
                jax.device_get(
                    jax.random.randint(thruster_key, shape=(), minval=0, maxval=self.nu)
                )
            )

        gp_perturbed_envs = self.get_perturbed_envs(env_key, 1.0, perturbed_envs)

        Perturbation.thruster_mask = Perturbation.thruster_mask.at[
            gp_perturbed_envs, jnp.full(gp_perturbed_envs.shape[0], thruster_index)
        ].set(self.failure_type.value)
        start_time_value = jnp.asarray(
            0.0 if start_time is None else start_time, dtype=self.start_times.dtype
        )
        self.start_times = self.start_times.at[
            gp_perturbed_envs, jnp.full(gp_perturbed_envs.shape[0], thruster_index)
        ].set(start_time_value)

        if not isinstance(valve_min, jnp.ndarray):
            valve_min = 0.15 * self.thruster_list[thruster_index].ctrlrange[-1]

        if not isinstance(valve_max, jnp.ndarray):
            valve_max = 0.8 * self.thruster_list[thruster_index].ctrlrange[-1]

        if valve_min is not None and valve_max is not None:
            gp_data_key = self._split_keys(None, 1)[0]
            x_data, y_data = ThrusterFailureSimulator(
                upper_bound=self.thruster_list[thruster_index].ctrlrange[-1],
                valve_min=valve_min,
                valve_max=valve_max,
            ).generate_failure_data(gp_data_key, self.failure_type)
        else:
            x_data = jnp.linspace(
                0.0,
                self.thruster_list[thruster_index].ctrlrange[-1],
                2,
            )
            y_data = x_data

        print(
            f"Thruster(s) are affected by a faulty valve starting at {start_time} seconds."
        )

        self.state = gp_register_state(
            self.state,
            Perturbation.thruster_mask,
            self.start_times,
            self._key,
            self.failure_type.value,
            thruster_index,
            jnp.asarray(x_data),
            jnp.asarray(y_data),
        )

    def key_callback(self, keycode: int | None = None) -> None:
        pass


class FaultyValve(GPPerturbation):
    def __init__(
        self,
        env_config: BaseEnvConfig,
        model_config: BaseModelConfig,
        key: jnp.ndarray,
    ) -> None:
        super().__init__(env_config, model_config, key, PerturbationStatus.FAULTY_VALVE)

    def register_perturbation(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None = None,
        index: int | None = None,
        start_time: float | None = None,
        valve_min: float | None = None,
        valve_max: float | None = None,
    ) -> None:
        if valve_min is None or valve_max is None:
            warnings.warn(
                "No min or max value set for the FaultyValve perturbation.", UserWarning
            )
        return super().register_perturbation(
            key, perturbed_envs, index, start_time, valve_min, valve_max
        )


class SaturatedThrust(GPPerturbation):
    def __init__(
        self,
        env_config: BaseEnvConfig,
        model_config: BaseModelConfig,
        key: jnp.ndarray,
    ) -> None:
        super().__init__(
            env_config, model_config, key, PerturbationStatus.SATURATED_THRUST
        )

    def register_perturbation(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None = None,
        index: int | None = None,
        start_time: float | None = None,
        valve_min: float | None = None,
        valve_max: float | None = None,
    ) -> None:
        return super().register_perturbation(
            key, perturbed_envs, index, start_time, valve_min, valve_max
        )


class ThrustInstability(GPPerturbation):
    def __init__(
        self,
        env_config: BaseEnvConfig,
        model_config: BaseModelConfig,
        key: jnp.ndarray,
    ) -> None:
        super().__init__(
            env_config, model_config, key, PerturbationStatus.THRUST_INSTABILITY
        )

    def register_perturbation(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None = None,
        index: int | None = None,
        start_time: float | None = None,
        valve_min: float | None = None,
        valve_max: float | None = None,
    ) -> None:
        return super().register_perturbation(
            key, perturbed_envs, index, start_time, valve_min, valve_max
        )
