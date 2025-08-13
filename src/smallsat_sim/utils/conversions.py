import numpy as np

def quaternion_from_yaw(yaw):
    """
    Computes a quaternion representing a rotation about the Z-axis (yaw).

    Parameters:
    ----------
    yaw : float
        The yaw angle (in radians) to convert into a quaternion.

    Returns:
    -------
    np.ndarray
        A quaternion [qw, qx, qy, qz] corresponding to the given yaw angle, where:
        - qx, qy are 0 (no roll or pitch assumed).
        - qz and qw encode the yaw rotation.
    """
    # Compute quaternion components for yaw (roll and pitch are zero)
    qw = np.cos(yaw / 2)
    qz = np.sin(yaw / 2)
    qx = 0.0
    qy = 0.0

    return np.array([qw, qx, qy, qz])


def yaw_from_quaternion(quat):
    """
    Computes the yaw (rotation about the Z-axis) from a quaternion.

    Parameters:
    ----------
    quat : np.ndarray or list
        A quaternion represented as [qw, qx, qy, qz], where:
        - qx, qy, qz are the vector components.
        - qw is the scalar component.

    Returns:
    -------
    float
        The yaw angle (in radians) corresponding to the rotation about the Z-axis.
    """
    qw, qx, qy, qz = quat  # Unpack quaternion components

    # Compute yaw
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy**2 + qz**2))

    return yaw

def convert_6dof_3dof(state_6dof: np.ndarray):
    """
    Converts a 6-DOF state vector to a reduced 3-DOF representation.

    This function reduces a full 6-DOF state, including position, orientation 
    (as a quaternion), linear velocity, and angular velocity, into a simplified 
    3-DOF state consisting of:
    - x and y position
    - yaw angle (rotation about the Z-axis)
    - x and y components of linear velocity
    - z component of angular velocity

    Parameters:
    ----------
    state_6dof : np.ndarray
        A numpy array representing the full 6-DOF state in the format:
        [pos_x, pos_y, pos_z, qw, qx, qy, qz, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z]

    Returns:
    -------
    np.ndarray
        A numpy array representing the reduced 3-DOF state in the format:
        [pos_x, pos_y, yaw, lin_vel_x, lin_vel_y, ang_vel_z]

        - pos_x, pos_y: x and y position of the object
        - yaw: rotation about the Z-axis (in radians)
        - lin_vel_x, lin_vel_y: x and y components of linear velocity
        - ang_vel_z: angular velocity about the Z-axis
    """
    pos = state_6dof[:3]
    quat = state_6dof[3:7]
    lin_vel = state_6dof[7:10]
    ang_vel = state_6dof[10:13]

    yaw = yaw_from_quaternion(quat)
    
    return np.array([
        pos[0],  # x position
        pos[1],  # y position
        yaw,     # yaw (rotation about Z-axis)
        lin_vel[0],  # x component of linear velocity
        lin_vel[1],  # y component of linear velocity
        ang_vel[2],  # z component of angular velocity
    ])


def convert_3dof_6dof(state_3dof: np.ndarray):
    """
    Reconstruct the full 6-DOF state from the reduced 3-DOF state.

    Parameters:
    state_3dof (np.ndarray): Reduced 3-DOF state as [x, y, yaw, lin_vel_x, lin_vel_y, ang_vel_z].

    Returns:
    np.ndarray: Full 6-DOF state as [pos_x, pos_y, pos_z, qw, qx, qy, qz, lin_vel_x, lin_vel_y, lin_vel_z, ang_vel_x, ang_vel_y, ang_vel_z].
    """
    x, y, yaw, lin_vel_x, lin_vel_y, ang_vel_z = state_3dof

    # Reconstruct position (z assumed to be 0)
    pos = np.array([x, y, 0.0])

    # Reconstruct quaternion from yaw (assuming roll=0, pitch=0)
    quat = quaternion_from_yaw(yaw)

    # Reconstruct linear velocity (z assumed to be 0)
    lin_vel = np.array([lin_vel_x, lin_vel_y, 0.0])

    # Reconstruct angular velocity (x, y assumed to be 0)
    ang_vel = np.array([0.0, 0.0, ang_vel_z])

    # Combine all components into the full 6-DOF state
    state_6dof = np.concatenate([pos, quat, lin_vel, ang_vel])
    return state_6dof
