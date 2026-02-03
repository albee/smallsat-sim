# This file includes various helper functions
# Can be included at the beginning of a file the following way:
# from utils.helpers import "function name"

# Parsing
import argparse
import numpy as np
import jax.numpy as jnp
import scipy.signal

import torch
import psutil
import os

from typing import Callable, TypeVar
from scipy.spatial.transform import Rotation as R

# Import all base classes for typing
from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.controllers.base_mpc_controller import BaseMPCController
from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.planners.base_planner import BasePlanner


T = TypeVar("T", np.ndarray, jnp.ndarray)


def get_args() -> argparse.Namespace:
    """
    This parser includes all non-environment and non-controller specific settings
    """
    # Create the parser
    parser = argparse.ArgumentParser(description="Parse command line inputs")

    # Add arguments
    parser.add_argument("--headless", action="store_true", help="Run in headless mode")
    parser.add_argument(
        "--num_bodies", type=int, help="Number of bodies in the simulation", default=1
    )
    parser.add_argument(
        "--video", action="store_true", help="Create a video of the experiment"
    )
    parser.add_argument("--log", action="store_true", help="Enable data logging")
    parser.add_argument(
        "--wandb", action="store_true", help="Use Weights & Biases to log RL data"
    )

    # Parse the arguments
    args = parser.parse_args()

    return args


def refModel3(x_d, v_d, a_d, r, wn_d, zeta_d, v_max, sampleTime):
    """[x_d,v_d,a_d] = refModel3(x_d,v_d,a_d,r,wn_d,zeta_d,v_max,sampleTime)
    is a 3-order reference model for generation of a smooth desired
    position x_d, velocity |v_d| < v_max, and acceleration a_d.
    Inputs are natural frequency wn_d and relative damping zeta_d.
    """
    # desired "jerk"
    j_d = (
        wn_d**3 * (r - x_d)
        - (2 * zeta_d + 1) * wn_d**2 * v_d
        - (2 * zeta_d + 1) * wn_d * a_d
    )

    # Forward Euler integration
    x_d += sampleTime * v_d  # desired position
    v_d += sampleTime * a_d  # desired velocity
    a_d += sampleTime * j_d  # desired acceleration

    # Limit the desired velocity
    np.clip(v_d, -v_max, v_max, out=v_d)

    return x_d, v_d, a_d


def Tquat(q: np.ndarray) -> np.ndarray:
    """Tq = Tquat(q) computes the quaternion transformation matrix Tq of
    dimension 4 x 3 for attitude such that q_dot = Tq * w
    """
    if len(q) == 4:
        eta = q[0]
        eps1 = q[1]
        eps2 = q[2]
        eps3 = q[3]

        T = 0.5 * np.array(
            [
                [-eps1, -eps2, -eps3],
                [eta, -eps3, eps2],
                [eps3, eta, -eps1],
                [-eps2, eps1, eta],
            ]
        )

    else:
        raise ValueError("input must be of dim. 4 (unit quaternion)")
    return T


def Rquat(q: np.ndarray) -> np.ndarray:
    """R = Rquat(q) computes the rotation matrix R of dimension 3 x 3
    for attitude from a quaternion q.
    """
    q = q.flatten()
    if len(q) == 4:
        eta = q[0]
        eps = q[1:4]

        S = skew(eps)
        R = np.eye(3) + 2 * eta * S + 2 * S @ S

    else:
        raise ValueError("input must be of dim. 4 (unit quaternion)")
    return R


def skew(x: np.ndarray) -> np.ndarray:
    return np.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def sgn_quat(x: float) -> int:
    """sgn = sgn_quat(x) returns the sign of a quaternion x."""
    if x >= 0:
        sgn = 1
    else:
        sgn = -1
    return sgn


def quat_multiply(q1: T, q2: T) -> T:
    """q = quat_multiply(q1,q2) computes the quaternion product q of
    two quaternions q1 and q2.
    """
    if len(q1) == 4 and len(q2) == 4:
        eta1 = q1[0]
        eps1_1 = q1[1]
        eps1_2 = q1[2]
        eps1_3 = q1[3]

        eta2 = q2[0]
        eps2_1 = q2[1]
        eps2_2 = q2[2]
        eps2_3 = q2[3]

        q = np.array(
            [
                eta1 * eta2 - eps1_1 * eps2_1 - eps1_2 * eps2_2 - eps1_3 * eps2_3,
                eta1 * eps2_1 + eps1_1 * eta2 + eps1_2 * eps2_3 - eps1_3 * eps2_2,
                eta1 * eps2_2 - eps1_1 * eps2_3 + eps1_2 * eta2 + eps1_3 * eps2_1,
                eta1 * eps2_3 + eps1_1 * eps2_2 - eps1_2 * eps2_1 + eps1_3 * eta2,
            ]
        )

    else:
        raise ValueError("input must be of dim. 4 (unit quaternion)")
    
    return q


def quat_conjugate(q) -> np.ndarray:
    """q_conj = quat_conjugate(q) computes the quaternion conjugate
    q_conj of a quaternion q.
    """
    if len(q) == 4:
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    else:
        raise ValueError("input must be of dim. 4 (unit quaternion)")
    return q_conj


def discount_cumsum(x, discount) -> np.ndarray:
    """
    Compute cumulative sums of vectors. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """
    return np.asarray(
        scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]
    )


def combined_shape(len, shape=None):
    """
    Combine two array shapes. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """
    if shape is None:
        return (len,)

    return (len, shape) if np.isscalar(shape) else (len, *shape)


def calc_lateral_tracking_error(obs: T, planner: BasePlanner) -> float:
    """
    Computes the lateral tracking error at a given point
    """
    current_pos = obs[:3]

    # Compute the closest point
    closest_point, _ = planner.closest_point_on_trajectory(point=obs[:3])

    # Compute l2 distance (is orthogonal already)
    return np.linalg.norm(closest_point - current_pos).item()


def calc_attitude_error(q_ref: T, q: T, w_first: bool = True) -> float:
    """
    Computes the attitude error as rotation angle.
    The angle error is the smallest angle by which you would need to rotate
    the object (or frame of reference) from its current orientation (actual quaternion)
    to match the desired orientation (desired quaternion).
    Returned in radians. Use np.degrees() for conversion.
    """
    # Normalize the quaternions to ensure they represent valid rotations.
    # Gracefully handle degenerate (zero-norm) quaternions which can occur during
    # initialization or faulty sensor readings.
    eps = 1e-12
    q_ref_norm = np.linalg.norm(q_ref)
    if q_ref_norm < eps:
        q_ref_normalized = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    else:
        q_ref_normalized = q_ref / q_ref_norm

    q_norm = np.linalg.norm(q)
    if q_norm < eps:
        q_normalized = np.array([0.0, 0.0, 0.0, 1.0], dtype=float)
    else:
        q_normalized = q / q_norm

    # Ensure the quaternions are in the format [x, y, z, w]
    # MuJoCo state quaternions are [w, x, y, z] by default.
    if w_first:
        q_ref_normalized = q_ref_normalized[[1, 2, 3, 0]]
        q_normalized = q_normalized[[1, 2, 3, 0]]

    # Adjust quaternion signs for continuity
    if np.dot(q_ref_normalized, q_normalized) < 0:
        q_normalized = -q_normalized

    # Convert the quaternions to Rotation objects
    rot_ref = R.from_quat(q_ref_normalized)
    rot = R.from_quat(q_normalized)

    # Compute the relative rotation from q to q_ref
    error_rotation = rot.inv() * rot_ref

    # Get the rotation vector (axis-angle representation)
    error_rotvec = error_rotation.as_rotvec()

    # Compute the angle (magnitude of the rotation vector)
    error_angle = np.linalg.norm(error_rotvec)

    # Ensure the angle is within [0, pi]
    if error_angle > np.pi:
        error_angle = 2 * np.pi - error_angle

    # Convert to degrees if requested
    error_angle = np.degrees(error_angle)

    return error_angle


def calc_model_error(
    obs: np.ndarray, x_past: np.ndarray, u_past: np.ndarray, f_int: Callable
) -> np.ndarray:

    model_error = (
        torch.from_numpy(obs - f_int(x_past, u_past).squeeze(-1))
        .to(torch.float64)
        .unsqueeze(-1)
    )

    return model_error.numpy()


def get_memory_usage():
    """
    Returns current memory usage of Python (on CPU)
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
