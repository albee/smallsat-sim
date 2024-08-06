# This file includes various helper functions
# Can be included at the beginning of a file the following way:
# from utils.helpers import "function name"

# Parsing
import argparse
import numpy as np
import torch
import psutil
import os

from typing import Callable

def get_args() -> argparse.Namespace:
    """
    This parser includes all non-environment and non-controller specific settings
    """
    # Create the parser
    parser = argparse.ArgumentParser(description='Parse command line inputs')

    # Add arguments
    parser.add_argument('--num_envs', type=int, help='Number of envs run in parallel', default=1)
    parser.add_argument('--headless', action='store_true', help='Run in headless mode')
    parser.add_argument('--num_bodies', type=int, help='Number of bodies in the simulation', default=1)
    parser.add_argument('--video', action='store_true', help='Create a video of the experiment')

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
    j_d = (wn_d**3 * (r -x_d) 
        - (2*zeta_d+1) * wn_d**2 * v_d 
        - (2*zeta_d+1) * wn_d * a_d)

    # Forward Euler integration
    x_d += sampleTime * v_d  # desired position
    v_d += sampleTime * a_d  # desired velocity
    a_d += sampleTime * j_d  # desired acceleration 
    
    # Limit the desired velocity
    np.clip(v_d, -v_max, v_max, out=v_d)

    return x_d, v_d, a_d


def Tquat(q) -> np.ndarray:
    """Tq = Tquat(q) computes the quaternion transformation matrix Tq of 
    dimension 4 x 3 for attitude such that q_dot = Tq * w 
    """
    if len(q) == 4:
        eta  = q[0]
        eps1 = q[1]
        eps2 = q[2]
        eps3 = q[3]
        
        T = 0.5 * np.array([[-eps1, -eps2, -eps3],
                           [eta, -eps3, eps2],
                           [eps3, eta, -eps1],
                           [-eps2, eps1, eta]])
        
    else:
        raise ValueError('input must be of dim. 4 (unit quaternion)')
    return T


def Rquat(q) -> np.ndarray:
    """R = Rquat(q) computes the rotation matrix R of dimension 3 x 3 
    for attitude from a quaternion q. 
    """
    q = q.flatten()
    if len(q) == 4:
        eta = q[0]
        eps = q[1:4]

        S = skew(eps)
        R = np.eye(3) + 2*eta*S + 2*S@S
        
    else:
        raise ValueError('input must be of dim. 4 (unit quaternion)')
    return R


def skew(x) -> np.ndarray:
    return np.array([[0, -x[2], x[1]],
                     [x[2], 0, -x[0]],
                     [-x[1], x[0], 0]])


def sgn_quat(x) -> int:
    """sgn = sgn_quat(x) returns the sign of a quaternion x. 
    """
    if x >= 0:
        sgn = 1
    else:
        sgn = -1
    return sgn


def quat_multiply(q1, q2) -> np.ndarray:
    """q = quat_multiply(q1,q2) computes the quaternion product q of 
    two quaternions q1 and q2. 
    """
    if len(q1) == 4 and len(q2) == 4:
        eta1   = q1[0]
        eps1_1 = q1[1]
        eps1_2 = q1[2]
        eps1_3 = q1[3]
        
        eta2   = q2[0]
        eps2_1 = q2[1]
        eps2_2 = q2[2]
        eps2_3 = q2[3]
        
        q = np.array([
            eta1*eta2   - eps1_1*eps2_1 - eps1_2*eps2_2 - eps1_3*eps2_3, 
            eta1*eps2_1 + eps1_1*eta2   + eps1_2*eps2_3 - eps1_3*eps2_2,
            eta1*eps2_2 - eps1_1*eps2_3 + eps1_2*eta2   + eps1_3*eps2_1,
            eta1*eps2_3 + eps1_1*eps2_2 - eps1_2*eps2_1 + eps1_3*eta2
            ])
        
    else:
        raise ValueError('input must be of dim. 4 (unit quaternion)')
    return q


def quat_conjugate(q) -> np.ndarray:
    """q_conj = quat_conjugate(q) computes the quaternion conjugate 
    q_conj of a quaternion q. 
    """
    if len(q) == 4:
        q_conj = np.array([q[0], -q[1], -q[2], -q[3]])
    else:
        raise ValueError('input must be of dim. 4 (unit quaternion)')
    return q_conj

def calc_model_error(obs: np.ndarray, x_past: np.ndarray, u_past: np.ndarray, f_int: Callable) -> np.ndarray:
    
    model_error = (
            torch.from_numpy(obs - f_int(x_past, u_past).squeeze(-1))
            .to(torch.float64)
            .unsqueeze(-1)
        )
    
    return model_error

def get_memory_usage():
    """
    Returns current memory usage of Python (on CPU)
    """
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    return mem_info.rss
