import numpy as np

from smallsat_sim.envs.dynamics_2d import SymbolicModel2D
from smallsat_sim.controllers.mpcc.controller_2d import NominalMPCCController2D
from smallsat_sim.envs.smallsat_hardware_2d.cfg.config import EnvConfig as SmallsatHardwareEnvCfg
from smallsat_sim.model.smallsat_hardware_2d.cfg.config import ModelConfig as SmallsatHardwareModelCfg
from smallsat_sim.envs.standalone_env import StandaloneEnv
from smallsat_sim.utils.conversions import convert_6dof_3dof, yaw_from_quaternion
# Different path options
from smallsat_sim.planners.mission.mission_2d_square import MissionPlanner2DSquare
from smallsat_sim.planners.mission.mission_2d_square_small import MissionPlanner2DSquareSmall 
from smallsat_sim.planners.mission.mission_2d_x_line import MissionPlanner2DxLine
from smallsat_sim.planners.mission.mission_2d_y_line import MissionPlanner2DyLine
from smallsat_sim.planners.mission.mission_2d_rot import MissionPlanner2Drot
import socket
import time
import argparse
import struct
import math
from collections import namedtuple
import csv
from datetime import datetime
import pytz
import os

# Define constants (adjust based on your system)
# BUF_DATA_SIZE = 21  # Example size, update if needed
BUF_DATA_SIZE = 132  # Example size, update if needed
BUF_DATA_HEADER_SIZE = 4  # Example header size
U32_BYTE_SIZE = 4  # Each float is 4 bytes
BUF_DATA_DIFFERENTIATOR = 99  # Example threshold value
BUF_DATA_UPPER_START = 100.0  # Placeholder value
BUF_DATA_LOWER_START = 0.0  # Placeholder value
BUF_DATA_HUNDRED_UNIT = 100.0  # Scaling factor
SEC_TO_MICROSEC_UNIT = 1000000
NUM_VARIABLES = 16

LOG_DATA = True
LOCAL_TESTING = False
MISSION_PLAN = "y_line" # Options: x_line, y_line, square_small, square

print("----------------------------")
print(f"Logging data: {LOG_DATA}")
print(f"Mission plan: {MISSION_PLAN}")
print("----------------------------")
# ---------------- LOG DATA ----------------
if LOG_DATA:
    timezone = pytz.timezone('America/Los_Angeles')
    now = datetime.now(timezone)
    date = now.strftime("%Y-%m-%d")
    current_time = now.strftime("%H:%M:%S %Z%z")   
    os.makedirs(f"logs/hardware/{date}", exist_ok=True)  # Creates folder if it doesn't exist
    # Generate filename with date and time
    file_path = os.path.join(f"logs/hardware/{date}", f"{current_time}.csv")

   # Column headers
    headers = ["Timestamp", "obs_pos_x", "obs_pos_y", "obs_pos_z", "obs_euler_roll", "obs_euler_pitch", "obs_euler_yaw", "obs_quat_w", "obs_quat_x", "obs_quat_y", "obs_quat_z", "obs_vel_x", "obs_vel_y", "obs_vel_z", "obs_rot_roll", "obs_rot_pitch", "obs_rot_yaw", "ref_x", "ref_y", "ref_att", "control_input", "solve_time", "control_callback_time"]

    # Write headers if file is empty
    with open(file_path, mode="a", newline="") as file:
        writer = csv.writer(file)
        if file.tell() == 0:  # Check if file is empty
            writer.writerow(headers)


# ---------------- PROCESS VICON DATA ----------------
# Create a structure for the processed Vicon state
ViconState = namedtuple("ViconState", ["timestamp", "position", "velocity", "quaternion", "euler_angles", "angular_velocity", "is_valid", "is_fresh"])

def process_vicon_data(buffer, start_recv_time_sec, curr_time_sec):
    """Processes raw Vicon UDP buffer data and extracts pose and velocity."""
    
    buf_data = buffer  # Assuming `buffer` is already a `bytes` object
    buf_size = len(buf_data)

    if buf_size != BUF_DATA_SIZE:
        print(f"Warning: Unexpected buffer size! Expected {BUF_DATA_SIZE}, got {buf_size}")
        return None

    # Compute timestamp relative to startRecvTimeSec
    timestamp = curr_time_sec - start_recv_time_sec

    #NUM_VARIABLES = (buf_size - BUF_DATA_HEADER_SIZE) # U32_BYTE_SIZE
    vicon_data = [0.0] * NUM_VARIABLES

    for i in range(NUM_VARIABLES):
        byte_offset = BUF_DATA_HEADER_SIZE + (i * U32_BYTE_SIZE)
        raw_value = buf_data[byte_offset]

        # Determine whether to start at upper or lower bound
        if raw_value >= BUF_DATA_DIFFERENTIATOR:
            vicon_data[i] = BUF_DATA_UPPER_START
            for j in range(U32_BYTE_SIZE):
                vicon_data[i] -= buf_data[byte_offset + j] / (BUF_DATA_HUNDRED_UNIT ** j)
        else:
            vicon_data[i] = BUF_DATA_LOWER_START
            for j in range(U32_BYTE_SIZE):
                vicon_data[i] += buf_data[byte_offset + j] / (BUF_DATA_HUNDRED_UNIT ** j)

    # Extract pose and velocity
    position = tuple(vicon_data[0:3])
    velocity = tuple(vicon_data[3:6])
    quaternion = tuple(vicon_data[6:10])
    euler_angles = tuple(vicon_data[10:13])
    angular_velocity = tuple(vicon_data[13:16])

    # Create Vicon state object
    vicon_state = ViconState(
        timestamp=timestamp,
        position=position,
        velocity=velocity,
        quaternion=quaternion,
        euler_angles=euler_angles,
        angular_velocity=angular_velocity,
        is_valid=True,
        is_fresh=True
    )

    return vicon_state

# ---------------- SETUP UDP ----------------
start_recv_time_sec = time.time()
parser = argparse.ArgumentParser(description='Description of your program')
# Set the IP and port for VOXL to listen on
UDP_PORT = 1153
SEND_UDP_PORT = 5005

# Local Testing PC: docker run -it --rm -v .:/code -p "1153:1153/udp" --net smallsat_hfnet --ip 127.0.0.1 smallsat:latest
# IP = "127.0.0.1" # Listen on all interfaces. 
# SEND_IP = "localhost" # IP

# Local Testing Voxl
if LOCAL_TESTING:
    IP = "127.0.0.1" # VOXL IP
    SEND_IP = "127.0.0.1" # PI IP

# Harware Testing
else:
    IP = "127.0.0.1" # VOXL IP
    SEND_IP = "127.0.0.1" # PI IP


# If receiving from boradcast: Listen on all available interfaces 0.0.0.0 IP
# parser.add_argument('--ip', help='UDP IP', default="127.0.0.1")
parser.add_argument('--ip', help='UDP IP', default=IP) # Voxl IP
parser.add_argument('--port', help='UDP PORT', default=UDP_PORT, type=int)
args = parser.parse_args()

# Create a socket
sock_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_receive.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 1)
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# If receiving from unicast
print(f"IP is: {args.ip}, Port is: {args.port}")
sock_receive.bind((args.ip, args.port))
print(f"Listening for Vicon data on {args.ip}:{args.port}...")

# ---------------- RUN CONTROLLER ----------------
cfg = SmallsatHardwareModelCfg()
model = SymbolicModel2D(cfg)
env = StandaloneEnv(env_cfg=SmallsatHardwareEnvCfg(), model=model)
env.model_cfg = cfg
Ts = env.env_cfg.control.NominalMPCC.Ts

# Create planner
if MISSION_PLAN == "square":
    planner = MissionPlanner2DSquare(env)
elif MISSION_PLAN == "square_small":
    planner = MissionPlanner2DSquareSmall(env)
elif MISSION_PLAN == "x_line":
    planner = MissionPlanner2DxLine(env)
elif MISSION_PLAN == "y_line":
    planner = MissionPlanner2DyLine(env)
elif MISSION_PLAN == "rot":
    planner = MissionPlanner2Drot(env)
else:
    print("No mission planner defined. Defaulting to y_line")
    planner = MissionPlanner2DyLine(env)

env.set_obs(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))

ctrl = None

# Receive messages in a loop
try:
    while True:
        start = time.time()
        # Get data from VICON
        # print("Waiting for data from vicon...")
        data, addr = sock_receive.recvfrom(1024) # Buffer size
        # print(f"Got data from {addr}")

        if data is None:
            print("No data received. Waiting.")
            time.sleep(0.25)
            continue
        curr_time_sec = time.time()

        if LOCAL_TESTING:
            message = data.decode()  # Convert bytes to string
            vicon_state = list(map(float, message.split(",")))  # Convert to list of floats
        else:
            # Harware Testing
            # ViconState(timestamp=8.290358066558838, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 0.0),
            #             euler_angles=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0), is_valid=True, is_fresh=True)
            start_vicon_process = time.time()
            vicon_state = process_vicon_data(data, start_recv_time_sec, curr_time_sec)
            end_vicon_process = time.time()
            vicon_process_duration = end_vicon_process - start_vicon_process
            # print(f"Vicon processing time: {vicon_process_duration}")
            # if vicon_state:
            #     print("Vicon State: ", vicon_state)

        if vicon_state is None:
            print("No data received. Waiting.")
            time.sleep(0.25)
            continue

        if(data is not None):
            if LOCAL_TESTING:
                tx, ty, tz = vicon_state[0], vicon_state[1], vicon_state[2]
                vx, vy, vz = vicon_state[3], vicon_state[4], vicon_state[5]
                qx, qy, qz, qw = vicon_state[6], vicon_state[7], vicon_state[8], vicon_state[9]
                yaw, roll, pitch = vicon_state[10], vicon_state[11], vicon_state[12]
                yaw_rate, pitch_rate, roll_rate = vicon_state[13], vicon_state[14], vicon_state[15]
            else:
                # Harware Testing
                # translation, rotation (quaternion), velocity, angular velocity 
                tx, ty, tz = vicon_state[1][0], vicon_state[1][1], vicon_state[1][2]
                vx, vy, vz = vicon_state[2][0], vicon_state[2][1], vicon_state[2][2]
                # qw, qx, qy, qz = vicon_state[3][0], vicon_state[3][1], vicon_state[3][2], vicon_state[3][3]
                qx, qy, qz, qw = vicon_state[3][0], vicon_state[3][1], vicon_state[3][2], vicon_state[3][3]
                yaw, pitch, roll = vicon_state[4][0], vicon_state[4][1], vicon_state[4][2]
                yaw_rate, pitch_rate, roll_rate = vicon_state[5][0], vicon_state[5][1], vicon_state[5][2]
    
            # print(f"Position: ({tx}, {ty}, {tz}) m")
            # print(f"Linear Velocity: ({vx}, {vy}, {vz}) m/s")
            # print(f"Orientation: ({qx}, {qy}, {qz}, {qw}) (quaternion)")
            print(f"Euler Angles: ({yaw}, {pitch}, {roll}) rad")
            # print(f"Angular Velocity: ({yaw_rate}, {pitch_rate}, {roll_rate}) rad/s")
            yaw_from_quat = yaw_from_quaternion([qw, qx, qy, qz])
            # print(f"Yaw from quat: {yaw_from_quat}")
            e_yaw = ((yaw - 0)+math.pi) % (2*math.pi) - math.pi # Wrap to [0, 2π]
            # print(f'Yaw: {yaw}')
            # print(f'Yaw error: {e_yaw}')
            # Quaternion error
            q_conj = np.array([qw, -qx, -qy, -qz])
            q_des = [1, 0, 0, 0]
            e_q = np.array(
                [
                    q_des[0] * q_conj[0]
                    - q_des[1] * q_conj[1]
                    - q_des[2] * q_conj[2]
                    - q_des[3] * q_conj[3],
                    q_des[0] * q_conj[1]
                    + q_des[1] * q_conj[0]
                    + q_des[2] * q_conj[3]
                    - q_des[3] * q_conj[2],
                    q_des[0] * q_conj[2]
                    - q_des[1] * q_conj[3]
                    + q_des[2] * q_conj[0]
                    + q_des[3] * q_conj[1],
                    q_des[0] * q_conj[3]
                    + q_des[1] * q_conj[2]
                    - q_des[2] * q_conj[1]
                    + q_des[3] * q_conj[0],
                ]
            )

            # We only want to minimize eps part of error quaternion
            e_q_vec = e_q - [1, 0, 0, 0]

            # Pass pose data to controller logic. Note: Vicon uses qx, qy, qz, qw and state definition is qw, qx, qy, qz!!!
            r, q, v, omega = np.array([tx, ty, tz]), np.array([qw, qx, qy, qz]), np.array([vx, vy, vz]), np.array([1, 0, 0, yaw_rate])
            # Set observations received in enviroment for controller
            env.set_obs(np.concatenate([r,q]), v, omega)
            if ctrl is None:
                ctrl = NominalMPCCController2D(env, planner)
            
            quat_error = e_q_vec.T @ (ctrl.ctrl_cfg.cost.Q_q) @ e_q_vec
            # print(f"Quat error: {quat_error}")
        

            control_input = ctrl.get_control_input(env)
            control_input[6] = control_input[6] * 1.25
            control_input[7] = control_input[7] * 1.3
            # thr7 = 0.125, thr8 = 0.13
            # for i in range(8):
                # print(f"Control input Th{i+1}: {control_input[i]}")
            # Map control inputs to actual Thrusters
            thruster_input = control_input[0:8]
            if LOG_DATA:
                # obs = [r (3),q (4),v (3), --> in body/inertial frame, omega (3)]
                obs = env.get_obs()
                # print(f"obs: {obs}")
                obs_pos_x = tx
                obs_pos_y = ty
                obs_pos_z = tz
                obs_euler_pitch = pitch
                obs_euler_roll = roll
                obs_euler_yaw = yaw
                obs_quat_w = qw
                obs_quat_x = qx
                obs_quat_y = qy
                obs_quat_z = qz
                obs_vel_x = vx
                obs_vel_y = vy
                obs_vel_z = vz
                obs_rot_roll = roll_rate
                obs_rot_pitch = pitch_rate
                obs_rot_yaw = yaw_rate
                # print(f"obs_pos: {obs_pos_x, obs_pos_y, obs_pos_z}, obs_quat: {obs_quat_w, obs_quat_x, obs_quat_y, obs_quat_z}, obs_vel: {obs_vel_x, obs_vel_y, obs_vel_z}, obs_rot: {obs_rot_roll, obs_rot_pitch, obs_rot_yaw}")

                ref_pt_pos, ref_pt_att = planner.get_reference(obs)
                ref_x = ref_pt_pos[0]
                ref_y = ref_pt_pos[1]
                ref_att_yaw = yaw_from_quaternion(ref_pt_att)
                ref_att = ref_att_yaw
                solve_time = ctrl.solve_time
                ctrl_loop_duration = ctrl.ctrl_input_callback_time
                with open(file_path, mode="a", newline="") as file:
                    writer = csv.writer(file)
                    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")       
                    # ["Timestamp", "obs_pos", "obs_euler", "obs_quat", "obs_vel", "obs_rot", "control_input"]     
                    writer.writerow([timestamp, obs_pos_x, obs_pos_y, obs_pos_z, obs_euler_roll, obs_euler_pitch, obs_euler_yaw, obs_quat_w, obs_quat_x, obs_quat_y, obs_quat_z, obs_vel_x, obs_vel_y, obs_vel_z, obs_rot_roll, obs_rot_pitch, obs_rot_yaw, ref_x, ref_y, ref_att, control_input, solve_time, ctrl_loop_duration])

            for i in range(len(thruster_input)):
                if thruster_input[i] < 0.0: # cannot have negative thruster
                    thruster_input[i] = 0.0
            
            # print("Thruster input: ", thruster_input)

            # Send control input to Pi
            # print(f"Sending thruster input to: {SEND_IP}:{SEND_UDP_PORT}")
            message = struct.pack(f'{thruster_input.size}f', *thruster_input)
            sock_send.sendto(message, (SEND_IP, SEND_UDP_PORT))
            end = time.time()

            duration = end - start
            # print(f"Duration: {duration}")
            if (Ts - duration) < 0:
                print("Solve time of controller to high.")
                continue
            time.sleep(Ts - duration)

            # print("----------------------------------------------------")

    
except KeyboardInterrupt:
    # Save log if logging is enabled
    # env.logger.save_log()
    print("\nReceiver shutting down.")
finally:
    sock_receive.close()