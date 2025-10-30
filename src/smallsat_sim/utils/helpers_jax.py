import argparse
import jax
import jax.numpy as jnp
from flax import nnx


# Import all base classes for typing
from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.controllers.base_mpc_controller import BaseMPCController
from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.planners.base_planner import BasePlanner

from scipy.spatial.transform import Rotation as R


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
    v_d = jnp.clip(v_d, -v_max, v_max)

    return x_d, v_d, a_d


def Tquat(q: jnp.ndarray) -> jnp.ndarray:
    """Tq = Tquat(q) computes the quaternion transformation matrix Tq of
    dimension 4 x 3 for attitude such that q_dot = Tq * w
    """
    if len(q) == 4:
        eta = q[0]
        eps1 = q[1]
        eps2 = q[2]
        eps3 = q[3]

        T = 0.5 * jnp.array(
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


def Rquat(q: jnp.ndarray) -> jnp.ndarray:
    """R = Rquat(q) computes the rotation matrix R of dimension 3 x 3
    for attitude from a quaternion q.
    """
    q = q.flatten()
    if len(q) == 4:
        eta = q[0]
        eps = q[1:4]

        S = skew(eps)
        R = jnp.eye(3) + 2 * eta * S + 2 * S @ S

    else:
        raise ValueError("input must be of dim. 4 (unit quaternion)")
    return R


def skew(x: jnp.ndarray) -> jnp.ndarray:
    return jnp.array([[0, -x[2], x[1]], [x[2], 0, -x[0]], [-x[1], x[0], 0]])


def sgn_quat(x: float) -> int:
    """sgn = sgn_quat(x) returns the sign of a quaternion x."""
    if x >= 0:
        sgn = 1
    else:
        sgn = -1
    return sgn


def quat_multiply(q1: jnp.ndarray, q2: jnp.ndarray) -> jnp.ndarray:
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

        q = jnp.array(
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


def quat_conjugate(q) -> jnp.ndarray:
    """q_conj = quat_conjugate(q) computes the quaternion conjugate
    q_conj of a quaternion q.
    """
    if len(q) == 4:
        q_conj = jnp.array([q[0], -q[1], -q[2], -q[3]])
    else:
        raise ValueError("input must be of dim. 4 (unit quaternion)")
    return q_conj


def discount_cumsum(x, discount) -> jnp.ndarray:
    """
    JAX-friendly discounted cumulative sum.
    """
    discount = jnp.asarray(discount)

    def scan_fn(carry, val):
        carry = val + discount * carry
        return carry, carry

    init = jnp.zeros_like(x[0])
    _, out = jax.lax.scan(scan_fn, init, x[::-1])
    return out[::-1]


def combined_shape(len, shape=None):
    """
    Combine two array shapes. Inspired from https://spinningup.openai.com/en/latest/algorithms/vpg.html.
    """
    if shape is None:
        return (len,)

    return (len, shape) if jnp.isscalar(shape) else (len, *shape)


def calc_lateral_tracking_error(obs: jnp.ndarray, planner: BasePlanner) -> jnp.ndarray:
    """
    Computes the lateral tracking error at a given point
    """
    # Compute the closest point
    closest_points = planner.closest_point_on_trajectory(obs)

    # Compute L2 distance (is orthogonal already)
    return jnp.linalg.norm(closest_points - obs[:, :3])


def calc_attitude_error(
    obs: jnp.ndarray, q_ref: jnp.ndarray = jnp.array([1, 0, 0, 0])
) -> jnp.ndarray:
    """
    Computes the attitude error as the shortest rotation angle. The angle error is the smallest 
    angle by which you would need to rotate the spacecraft (or frame of reference) from its 
    current orientation (actual quaternion) to match the desired orientation (desired quaternion).
    Returned in radians. Use jnp.degrees() for conversion.
    """
    # Normalize the quaternions, defaulting to identity if the norm is near zero
    eps = 1e-12
    q_ref = jnp.asarray(q_ref)
    if q_ref.ndim == 1:
        q_ref = jnp.broadcast_to(q_ref, (obs.shape[0], q_ref.shape[0]))
    elif q_ref.shape[0] != obs.shape[0]:
        raise ValueError(
            "q_ref must be either a single quaternion or batched to match obs"
        )

    q_ref_norm = jnp.linalg.norm(q_ref, axis=1, keepdims=True)
    q_ref_norm = jnp.maximum(q_ref_norm, eps)
    q_ref_normalized = q_ref / q_ref_norm

    q = obs[:, 3:7]
    q_norm = jnp.linalg.norm(q, axis=1, keepdims=True)
    q_norm = jnp.maximum(q_norm, eps)
    q_normalized = q / q_norm

    # Compute the error quaternion using JAX operations
    # Compute the conjugate of q_normalized in a vectorized manner
    q_conj = jnp.concatenate([q_normalized[:, :1], -q_normalized[:, 1:]], axis=1)

    # Compute the error quaternion: q_err = q_ref_normalized * q_conj
    q_err = jax.vmap(quat_multiply)(q_ref_normalized, q_conj)
    q_err = q_err / jnp.linalg.norm(q_err, axis=1, keepdims=True)

    # For a quaternion q = [w, x, y, z], the rotation angle is given by 2*arccos(|w|)
    w = jnp.clip(q_err[:, 0], -1.0, 1.0)
    error_angle = 2 * jnp.arccos(jnp.abs(w))

    # Ensure the angle is within [0, pi]
    # error_angle = jnp.where(error_angle > jnp.pi, 2 * jnp.pi - error_angle, error_angle)

    v_norm = jnp.linalg.norm(q_err[:, 1:], axis=1)
    error_angle = 2 * jnp.arctan2(v_norm, jnp.abs(w))

    return error_angle


def calc_extrinsic_error(
    actual_ext: jnp.ndarray, estimated_ext: jnp.ndarray
) -> jnp.ndarray:
    """
    Computes the error between the actual and estimated wrenches in all environments.
    """
    return estimated_ext - actual_ext


def train_val_split(
    X,
    y,
    key: jnp.ndarray,
    val_split=0.2,
    shuffle: bool = True,
    trim_for_cnn: bool = False,
):
    """
    Split data into training and validation set.
    """
    num_samples = X.shape[0]
    key, subkey = jax.random.split(key)
    if shuffle:
        indices = jax.random.permutation(subkey, num_samples)
    else:
        indices = jnp.arange(num_samples)

    val_size = int(num_samples * val_split)
    train_idx, val_idx = indices[val_size:], indices[:val_size]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    if trim_for_cnn:
        # Trim the data to be a multiple of seq_len
        X_train, y_train = _trim_and_reshape(X_train, y_train)
        X_val, y_val = _trim_and_reshape(X_val, y_val)

    return X_train, y_train, X_val, y_val, key


def _trim_and_reshape(X, y, seq_len=50):
    """
    Trim leading dimension to a multiple of ``seq_len`` and construct sequences
    that stay within the same environment trajectory.
    """
    if X.shape[0] == 0:
        return X, y

    keep = (X.shape[0] // seq_len) * seq_len  # largest multiple of seq_len ≤ n
    X_trim = X[:keep]
    y_trim = y[:keep]

    if X_trim.ndim >= 3:
        num_envs = X_trim.shape[1]
        feat_dim = X_trim.shape[2]
        ext_dim = y_trim.shape[2] if y_trim.ndim >= 3 else y_trim.shape[-1]

        if keep == 0:
            return (
                jnp.empty((0, seq_len, feat_dim), dtype=X.dtype),
                jnp.empty((0, seq_len, ext_dim), dtype=y.dtype),
            )

        windows = keep // seq_len
        X_windows = X_trim.reshape(windows, seq_len, num_envs, feat_dim)
        X_windows = jnp.transpose(X_windows, (2, 0, 1, 3))  # env, window, seq, feat
        X_seq = X_windows.reshape(-1, seq_len, feat_dim)

        if y_trim.ndim >= 3:
            y_windows = y_trim.reshape(windows, seq_len, num_envs, y_trim.shape[2])
            y_windows = jnp.transpose(y_windows, (2, 0, 1, 3))
            y_seq = y_windows.reshape(-1, seq_len, y_trim.shape[2])
        else:
            y_windows = y_trim.reshape(windows, seq_len, num_envs)
            y_windows = jnp.transpose(y_windows, (2, 0, 1))
            y_seq = y_windows.reshape(-1, seq_len, 1)
    else:
        if keep == 0:
            return (
                jnp.empty((0, seq_len, X.shape[-1]), dtype=X.dtype),
                jnp.empty((0, seq_len, y.shape[-1]), dtype=y.dtype),
            )
        X_seq = X_trim.reshape(-1, seq_len, X.shape[-1])
        y_seq = y_trim.reshape(-1, seq_len, y.shape[-1])

    return X_seq, y_seq


def batch_mse_loss_fn(model, X: jnp.ndarray, y: jnp.ndarray) -> jnp.ndarray:
    """
    Mean squared error loss function.
    """
    # TODO: implement sliding window for batch loss
    preds = jax.vmap(lambda x: model(x), in_axes=0)(X)
    losses = jnp.square(preds - y[:, -1, :])

    return jnp.mean(losses)


@nnx.jit
def mae_loss_fn(model, X: jnp.ndarray, y: jnp.ndarray, key) -> jnp.ndarray:
    """
    Mean squared error loss function.
    """
    y_pred_dist, _ = model.forward(X)
    y_pred = y_pred_dist.sample(seed=key)

    return jnp.mean(jnp.abs(y_pred - y))


def normalize_obs(obs: jnp.ndarray, ep_obs: jnp.ndarray, step: int) -> jnp.ndarray:
    """
    Normalize the observations along a trajectory.
    """
    return (obs - ep_obs[:, : step + 1, :].mean(axis=1)) / (
        ep_obs[:, : step + 1, :].std(axis=1) + 1e-8
    )


def scale_rews(rews: jnp.ndarray, ep_rets: jnp.ndarray, step: int) -> jnp.ndarray:
    """
    Scale the rewards along a trajectory.
    """
    return rews / (ep_rets[:, : step + 1].std(axis=1) + 1e-8)
