from abc import ABC, abstractmethod
from dataclasses import replace
import warnings
from copy import deepcopy
from typing import Optional

import jax
import jax.numpy as jnp

from smallsat_sim.envs.base_env_config import BaseEnvConfig
from smallsat_sim.envs.perturbation_gp import ThrusterFailureSimulator
from smallsat_sim.envs.perturbation_state import (
    PerturbationState,
    PerturbationStatus,
    _derive_gp_resolution,
    gp_apply_from_state,
    gp_register_state,
    perturbation_state_from_serializable,
    perturbation_state_to_serializable,
    stuck_off_activate_state,
    stuck_off_apply_from_state,
    stuck_on_activate_state,
    stuck_on_apply_from_state,
)
from smallsat_sim.model.base_model_config import BaseModelConfig




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
        self.verbose = getattr(env_config.sim, "verbose", False)
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

    def resolve_target_envs(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray | None,
    ) -> jnp.ndarray:
        """
        Resolve target environments for a failure registration.
        """
        return self.get_perturbed_envs(key, 1.0, perturbed_envs)

    def select_shared_operational_thruster(
        self,
        key: jnp.ndarray,
        perturbed_envs: jnp.ndarray,
        preferred_thruster: int | None = None,
    ) -> int | None:
        """
        Select one thruster index that is operational across all target environments.
        Returns `None` if no shared operational thruster exists.
        """
        if perturbed_envs.size == 0:
            return None

        selected_envs = Perturbation.thruster_mask[perturbed_envs]
        shared_operational = jnp.all(
            selected_envs == PerturbationStatus.OPERATIONAL.value, axis=0
        )
        valid_thrusters = jnp.where(
            shared_operational, size=self.nu, fill_value=-1
        )[0]
        valid_mask = valid_thrusters != -1
        num_valid = int(valid_mask.sum())
        if num_valid <= 0:
            return None

        if preferred_thruster is not None:
            preferred_idx = int(preferred_thruster)
            if 0 <= preferred_idx < self.nu and bool(shared_operational[preferred_idx]):
                return preferred_idx
            return None

        probs = valid_mask.astype(jnp.float32) / float(num_valid)
        chosen = jax.random.choice(key, valid_thrusters, p=probs)
        return int(jax.device_get(chosen))

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
        if getattr(self.perturbations[perturbation_idx], "verbose", False):
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
        self._gp_num_points, self._gp_subset_size = _derive_gp_resolution(
            self.nu, self.num_envs
        )
        self._gp_failure_mode_overrides: dict | None = None

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
        stuck_off_envs = self.resolve_target_envs(env_key, perturbed_envs)
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

        if self.verbose:
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

        self.min_thruster_force = jnp.asarray(
            [
                thruster.forcerange[0]
                for thruster in self.model_config.Thrusters.thruster_list
            ],
            dtype=self.start_times.dtype,
        )
        self.max_thruster_force = jnp.asarray(
            [
                thruster.forcerange[1]
                for thruster in self.model_config.Thrusters.thruster_list
            ],
            dtype=self.start_times.dtype,
        )
        # Per env/thruster stuck-on force value. Active channels are sampled uniformly
        # within each thruster's physical force range when failure is registered.
        self.stuck_on_force = jnp.tile(
            self.max_thruster_force[None, :], (self.num_envs, 1)
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
            max_thruster_force=self.stuck_on_force,
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
        env_key, thruster_key, force_key = jax.random.split(key, 3)
        stuck_on_envs = self.resolve_target_envs(env_key, perturbed_envs)

        stuck_on_thrusters = self.select_thrusters(
            thruster_key, stuck_on_envs, perturbed_thrusters
        )
        min_force = self.min_thruster_force[stuck_on_thrusters]
        max_force = self.max_thruster_force[stuck_on_thrusters]
        sampled_force = jax.random.uniform(
            force_key,
            shape=(stuck_on_envs.shape[0],),
            minval=min_force,
            maxval=max_force,
        )
        self.stuck_on_force = self.stuck_on_force.at[
            stuck_on_envs, stuck_on_thrusters
        ].set(sampled_force)

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
            self.stuck_on_force,
            self._key,
        )

        if self.verbose:
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
        (
            self._gp_num_points,
            self._gp_subset_size,
        ) = _derive_gp_resolution(self.nu, self.num_envs)
        self._gp_failure_mode_overrides: Optional[dict] = None
        self._gp_sample_cache: dict[tuple, tuple[jnp.ndarray, jnp.ndarray]] = {}

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

    def _gp_simulator_kwargs(self, **base_kwargs):
        kwargs = dict(base_kwargs)
        kwargs["num_points"] = self._gp_num_points
        kwargs["subset_size"] = self._gp_subset_size
        if self._gp_failure_mode_overrides:
            kwargs["failure_mode_overrides"] = self._gp_failure_mode_overrides
        return kwargs

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
        env_key, thruster_key = jax.random.split(key)
        gp_perturbed_envs = self.resolve_target_envs(env_key, perturbed_envs)
        if gp_perturbed_envs.size == 0:
            return

        thruster_index = self.select_shared_operational_thruster(
            thruster_key,
            gp_perturbed_envs,
            preferred_thruster=index,
        )
        if thruster_index is None:
            if self.verbose:
                print(
                    "Could not register GP perturbation: no shared operational thruster "
                    "for selected environments."
                )
            return

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
            simulator_kwargs = self._gp_simulator_kwargs(
                upper_bound=self.thruster_list[thruster_index].ctrlrange[-1],
                valve_min=valve_min,
                valve_max=valve_max,
            )
            cache_key = (
                int(thruster_index),
                float(simulator_kwargs["upper_bound"]),
                float(valve_min),
                float(valve_max),
                int(simulator_kwargs["num_points"]),
                int(simulator_kwargs["subset_size"]),
                int(self.failure_type.value),
            )
            cached = self._gp_sample_cache.get(cache_key)
            if cached is None:
                x_data, y_data = ThrusterFailureSimulator(
                    **simulator_kwargs
                ).generate_failure_data(gp_data_key, self.failure_type)
                self._gp_sample_cache[cache_key] = (x_data, y_data)
            else:
                x_data, y_data = cached
        else:
            x_data = jnp.linspace(
                0.0,
                self.thruster_list[thruster_index].ctrlrange[-1],
                2,
            )
            y_data = x_data

        if self.verbose:
            start_time_print = float(start_time_value)
            print(
                "Thruster(s) are affected by a faulty valve starting at "
                f"{start_time_print} seconds."
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
