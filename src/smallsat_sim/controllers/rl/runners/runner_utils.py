from collections.abc import Mapping, Sequence
from flax import nnx
from flax.nnx.variablelib import VariableState
from mujoco import mjx
import jax
import jax.numpy as jnp
import numpy as np
import pickle

from smallsat_sim.envs.disturbances import (
    DisturbanceState,
    disturbance_state_from_serializable,
    disturbance_state_to_serializable,
)
from smallsat_sim.envs.perturbations_rl import (
    PerturbationState,
    perturbation_state_from_serializable,
    perturbation_state_to_serializable,
)
from smallsat_sim.envs.vec_env import (
    VecEnvState,
    vecenv_state_from_serializable,
    vecenv_state_to_serializable,
)

_SERIALIZATION_TYPE_KEY = "__smallsat_type__"


def _is_jax_array(x):
    return isinstance(x, (jnp.ndarray, jax.Array))


def _prepare_for_pickle(obj):
    if isinstance(obj, VecEnvState):
        return {
            _SERIALIZATION_TYPE_KEY: "VecEnvState",
            "payload": vecenv_state_to_serializable(obj),
        }
    if isinstance(obj, DisturbanceState):
        return {
            _SERIALIZATION_TYPE_KEY: "DisturbanceState",
            "payload": disturbance_state_to_serializable(obj),
        }
    if isinstance(obj, PerturbationState):
        return {
            _SERIALIZATION_TYPE_KEY: "PerturbationState",
            "payload": perturbation_state_to_serializable(obj),
        }
    if _is_jax_array(obj):
        return np.asarray(obj)
    if isinstance(obj, np.ndarray):
        return obj
    if isinstance(obj, Mapping):
        return {k: _prepare_for_pickle(v) for k, v in obj.items()}
    if isinstance(obj, Sequence) and not isinstance(obj, (str, bytes)):
        converted = [_prepare_for_pickle(v) for v in obj]
        return type(obj)(converted) if not isinstance(obj, tuple) else tuple(converted)
    return obj


def _restore_from_serializable(obj, *, mjx_batch_template: mjx.Data | None = None):
    if isinstance(obj, Mapping):
        if _SERIALIZATION_TYPE_KEY in obj:
            payload = obj["payload"]
            kind = obj[_SERIALIZATION_TYPE_KEY]
            if kind == "VecEnvState":
                if mjx_batch_template is None:
                    return payload
                return vecenv_state_from_serializable(
                    payload, mjx_batch_template=mjx_batch_template
                )
            if kind == "DisturbanceState":
                return disturbance_state_from_serializable(payload)
            if kind == "PerturbationState":
                return perturbation_state_from_serializable(payload)
        return {
            k: _restore_from_serializable(v, mjx_batch_template=mjx_batch_template)
            for k, v in obj.items()
        }
    if isinstance(obj, list):
        return [
            _restore_from_serializable(v, mjx_batch_template=mjx_batch_template)
            for v in obj
        ]
    if isinstance(obj, tuple):
        return tuple(
            _restore_from_serializable(v, mjx_batch_template=mjx_batch_template)
            for v in obj
        )
    return obj


def _to_jnp_recursive(obj):
    if isinstance(obj, np.ndarray):
        return jnp.asarray(obj)
    if isinstance(obj, (VecEnvState, DisturbanceState, PerturbationState)):
        return obj
    if isinstance(obj, Mapping):
        return {k: _to_jnp_recursive(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_jnp_recursive(v) for v in obj]
    if isinstance(obj, tuple):
        return tuple(_to_jnp_recursive(v) for v in obj)
    return obj


def save_training_data(data_path: str, data_filename: str, data: dict) -> None:
    """
    Save the data obtained by interacting with the environment.
    """
    payload = _prepare_for_pickle(data)
    with open(data_path + data_filename, "wb") as file:
        pickle.dump(payload, file)
    print(f"Training data saved to {data_filename}")


def save_trained_modules(agent, ckpt_dir: str, ckpt_filename: str) -> None:
    """
    Save the actor and critic network params.
    """
    training_state = {
        "actor_model": nnx.state(agent.actor),
        "critic_model": nnx.state(agent.critic),
    }
    payload = _prepare_for_pickle(training_state)
    with open(ckpt_dir + ckpt_filename, "wb") as file:
        pickle.dump(payload, file)
    print(f"Checkpoint saved to {ckpt_filename}")


def save_adaptation_module(am, ckpt_dir: str, ckpt_filename: str) -> None:
    """
    Save the actor and critic network params.
    """
    training_state = {
        "am_model": nnx.state(am),
    }
    payload = _prepare_for_pickle(training_state)
    with open(ckpt_dir + ckpt_filename, "wb") as file:
        pickle.dump(payload, file)
    print(f"Checkpoint saved to {ckpt_filename}")


def load_training_data(
    data_path: str,
    data_filename: str,
    *,
    mjx_batch_template: mjx.Data | None = None,
):
    """
    Load the training data.
    """
    with open(data_path + data_filename, "rb") as file:
        raw = pickle.load(file)
    data = _restore_from_serializable(raw, mjx_batch_template=mjx_batch_template)
    data = _to_jnp_recursive(data)
    print(f"Checkpoint data loaded from {data_filename}")

    return data


def load_trained_modules(ckpt_dir: str, ckpt_filename: str):
    """
    Load the actor and critic network params.
    """
    with open(ckpt_dir + ckpt_filename, "rb") as file:
        raw = pickle.load(file)
    restored_state = _restore_from_serializable(raw)
    restored_state = _to_jnp_recursive(restored_state)
    print(f"Checkpoint loaded from {ckpt_filename}")

    return restored_state


def _copy_value_if_shape_matches(
    target_value: jnp.ndarray,
    source_value: jnp.ndarray,
) -> jnp.ndarray | None:
    target = jnp.asarray(target_value)
    source = jnp.asarray(source_value)
    if target.shape == source.shape:
        return source
    return None


def align_checkpoint_state_to_model(target_state, source_state):
    """
    Align a checkpoint state to a model state.

    Matching parameters are copied exactly. Non-matching leaves keep their
    target initialization so checkpoint loading does not silently create
    zero-padded context inputs.
    """
    if isinstance(target_state, VariableState) and isinstance(
        source_state, VariableState
    ):
        copied_value = _copy_value_if_shape_matches(
            target_state.value,
            source_state.value,
        )
        if copied_value is None:
            return target_state
        return target_state.replace(value=copied_value)

    if isinstance(target_state, Mapping) and isinstance(source_state, Mapping):
        aligned_items = {}
        for key, target_value in target_state.items():
            if key in source_state:
                aligned_items[key] = align_checkpoint_state_to_model(
                    target_value,
                    source_state[key],
                )
            else:
                aligned_items[key] = target_value
        return aligned_items

    if isinstance(target_state, np.ndarray) or _is_jax_array(target_state):
        if isinstance(source_state, np.ndarray) or _is_jax_array(source_state):
            copied_value = _copy_value_if_shape_matches(
                target_state,
                source_state,
            )
            if copied_value is not None:
                return copied_value
        return target_state

    return source_state if type(target_state) is type(source_state) else target_state


def update_module_from_checkpoint_state(module, checkpoint_state) -> None:
    """Update ``module`` from checkpoint state, expanding first-layer inputs if needed."""
    target_state = nnx.state(module)
    aligned_state = align_checkpoint_state_to_model(target_state, checkpoint_state)
    nnx.update(module, aligned_state)
