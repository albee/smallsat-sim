from smallsat_sim.controllers.base_controller import BaseController
from smallsat_sim.utils.helpers import quat_multiply, quat_conjugate, Rquat, sgn_quat
import yaml
import numpy as np

import mujoco


class PDController(BaseController):
    def __init__(self) -> None:
        super().__init__()
        mujoco.set_mjcb_control(self.controller_callback)

        self.x_ref = np.zeros((3,1))
        self.quat_ref = np.array([1.,0.,0.,0.])
        self.v_ref = np.zeros((3,1))
        self.omega_ref = np.zeros((3,1))  # Angular velocity
        cfg = self._load_cfg('pd')
        self.Kp_x = cfg['gains']['Kp_x']
        self.Kd_x = cfg['gains']['Kd_x']
        self.Kp_q = cfg['gains']['Kp_q']
        self.Kd_q = cfg['gains']['Kd_q']

    def B_matrix(self,model,data) -> np.ndarray:
        """Create B matrix (thruster configuration matrix, mixer, etc.)
        Not sure how to reflect that the thrusters only produce thrust 
        in one direction. For now, assume bi-directional thrusters.
        """
        B_matrix = np.empty(shape=(6,model.nu))
        for i in range(model.nu):
            actuator = model.actuator(i)
            force = model.actuator(i).gear[0:3]  # Force produced by thruster 'i'
            actuator_pos = model.site(i).pos  # Position of thruster 'i' relative to body origin
            # Add direct forces and torques to B matrix. Then calculate force arms and add
            B_matrix[:,i] = (actuator.gear 
                             +np.append([0,0,0],np.cross(force,actuator_pos)))
        return B_matrix

    def controller_callback(self,model,data) -> None:
        """Defines the controller callback for the simulation step.
        """
        desired_pos        = self.x_ref  # Linear
        desired_quat       = np.array([1.,0.,0.,0.])
        desired_linvel     = self.v_ref  # Linear
        desired_angvel     = np.zeros((3,1))
        current_pos        = np.reshape(data.qpos[:3],(3,1))
        current_quat       = np.reshape(data.qpos[3:],(4,1))
        current_linvel     = np.reshape(data.qvel[:3],(3,1))  # Linear
        current_angvel     = np.reshape(data.qvel[3:],(3,1))  # Angular

        x_error = desired_pos - current_pos  # Linear, world frame
        v_error = desired_linvel - current_linvel  # Linear, world frame
        # Desired quaternion - current quaternion:
        quat_error = quat_multiply(desired_quat, quat_conjugate(current_quat))
        eta_error = quat_error[0]
        eps_error = quat_error[1:]
        
        # PD, linear part. Note: both position and velocity are decomposed
        # in the world frame. R.T rotates them to the body frame
        R_WB = Rquat(current_quat)  # Rotation matrix from body to world
        desired_linacc = (self.Kp_x * R_WB.T @ x_error  # Desired linear acceleration
                         +self.Kd_x * R_WB.T @ v_error)
        # PD, angular part, (alpha=angular acceleration). 
        # Note: angular velocity is decomposed in the body frame
        desired_alpha = -(self.Kp_q * sgn_quat(eta_error) * eps_error
                         +self.Kd_q * (desired_angvel - current_angvel))
        desired_acceleration = np.append(desired_linacc, desired_alpha)

        desired_control = desired_acceleration
        # Distribute the desired forces and torques to the actuators, least squares
        B_matrix = self.B_matrix(model,data)
        data.ctrl = np.dot(np.linalg.pinv(B_matrix), desired_control)
