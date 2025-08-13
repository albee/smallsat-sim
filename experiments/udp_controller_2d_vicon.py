import numpy as np

from smallsat_sim.envs.dynamics_2d import SymbolicModel2D
from smallsat_sim.controllers.mpcc.controller_2d import NominalMPCCController2D
from smallsat_sim.envs.astrobee_2d.cfg.config import EnvConfig as AstroBeeEnvCfg
from smallsat_sim.model.astrobee_2d.cfg.config import ModelConfig as AstroBeeModelCfg
from smallsat_sim.envs.standalone_env import StandaloneEnv
from smallsat_sim.planners.mission.mission_2d import MissionPlanner2D
import socket
import time
import argparse
import struct
import math
from collections import namedtuple

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



start_recv_time_sec = time.time()
parser = argparse.ArgumentParser(description='Description of your program')
# Set the IP and port for VOXL to listen on
UDP_PORT = 1153
IP = "127.0.0.1" # VOXL IP

SEND_UDP_PORT = 5005
SEND_IP = "127.0.0.1" # PI IP
# SEND_IP = "255.255.255.255"

# If receiving from boradcast: Listen on all available interfaces 0.0.0.0 IP
# parser.add_argument('--ip', help='UDP IP', default="127.0.0.1")
parser.add_argument('--ip', help='UDP IP', default=IP) # Voxl IP
parser.add_argument('--port', help='UDP PORT', default=UDP_PORT, type=int)
args = parser.parse_args()

cfg = AstroBeeModelCfg()
model = SymbolicModel2D(cfg)
env = StandaloneEnv(env_cfg=AstroBeeEnvCfg(), model=model)
env.model_cfg = cfg
planner =  MissionPlanner2D(env) 
Ts = env.env_cfg.control.NominalMPCC.Ts

env.set_obs(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))

ctrl = None

# Create a socket
sock_receive = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
# Enable broadcast
# sock_send.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  
# If receiving from boradcast
# sock_receive.bind(("", args.port)) # listen to all available interfaces
# print(f"Listening for Vicon data on:{args.port}...")

# If receiving from unicast
print(f"IP is: {args.ip}, Port is: {args.port}")
sock_receive.bind((args.ip, args.port))
print(f"Listening for Vicon data on {args.ip}:{args.port}...")

# Receive messages in a loop
try:
    while True:
        start = time.time()
        # Get data from VICON
        print("Waiting for data from vicon...")
        data, addr = sock_receive.recvfrom(1024) # Buffer size
        print(f"Got data from {addr}")

        if data is None:
            print("No data received. Waiting.")
            time.sleep(0.25)
            continue
        curr_time_sec = time.time()

        # Harware Testing
        #  ViconState(timestamp=8.290358066558838, position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 0.0),
        #             euler_angles=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0), is_valid=True, is_fresh=True)
        # vicon_state = process_vicon_data(data, start_recv_time_sec, curr_time_sec)

        # Local testing
        message = data.decode()  # Convert bytes to string
        vicon_state = list(map(int, message.split(",")))  # Convert to list of integers

        if vicon_state:
            print("Vicon State: ", vicon_state)

        # if len(vicon_state) == 0:
        #     print("No data received. Waiting.")
        #     time.sleep(0.25)
        #     continue

        if vicon_state is None:
            print("No data received. Waiting.")
            time.sleep(0.25)
            continue

        if(data is not None):
            # Harware Testing
            # translation, rotation (quaternion), velocity, angular velocity 
            # tx, ty, tz = vicon_state[1][0], vicon_state[1][1], vicon_state[1][2]
            # vx, vy, vz = vicon_state[2][0], vicon_state[2][1], vicon_state[2][2]
            # qw, qx, qy, qz = vicon_state[3][0], vicon_state[3][1], vicon_state[3][2], vicon_state[3][3]
            # roll, pitch, yaw = vicon_state[4][0], vicon_state[4][1], vicon_state[4][2]
            # roll_rate, pitch_rate, yaw_rate = vicon_state[5][0], vicon_state[5][1], vicon_state[5][2]

            # Local Testing
            tx, ty, tz = vicon_state[0], vicon_state[1], vicon_state[2]
            vx, vy, vz = vicon_state[3], vicon_state[4], vicon_state[5]
            qw, qx, qy, qz = vicon_state[6], vicon_state[7], vicon_state[8], vicon_state[9]
            roll, pitch, yaw = vicon_state[10], vicon_state[11], vicon_state[12]
            roll_rate, pitch_rate, yaw_rate = vicon_state[13], vicon_state[14], vicon_state[15]
    
            print(f"Position: ({tx}, {ty}, {tz}) m")
            print(f"Linear Velocity: ({vx}, {vy}, {vz}) m/s")
            print(f"Orientation: ({qw}, {qx}, {qy},{qz}) (quaternion)")
            print(f"Euler Angles: ({roll}, {pitch}, {yaw}) rad")
            print(f"Angular Velocity: ({roll_rate}, {pitch_rate}, {yaw_rate}) rad/s")

            # Pass pose data to controller logic
            r, q, v, omega = np.array([tx, ty, tz]), np.array([qx, qy, qz, qw]), np.array([vx, vy, vz]), np.array([1, 0, 0, yaw_rate])
            # Set observations received in enviroment for controller
            env.set_obs(np.concatenate([r,q]), v, omega)
            if ctrl is None:
                ctrl = NominalMPCCController2D(env, planner)

            control_input = ctrl.get_control_input(env)
            print("Control input: ", control_input)
            # Map control inputs to actual Thrusters
            thruster_input = control_input[0:8]
            thruster_input[0] = control_input[0]
            thruster_input[1] = control_input[2]
            thruster_input[2] = control_input[4]
            thruster_input[3] = control_input[6]
            thruster_input[4] = control_input[1]
            thruster_input[5] = control_input[3]
            thruster_input[6] = control_input[5]
            thruster_input[7] = control_input[7]

            for i in range(len(thruster_input)):
                # if thruster_input[i] < 0.01:
                #     thruster_input[i] = 0.0
                if thruster_input[i] < 0.0: # cannot have negative thruster
                    thruster_input[i] = 0.0
            
            print("Thruster input: ", thruster_input)

            # Send control input to Pi
            # message = struct.pack('f', control_input)
            print(f"Sending thruster input to: {SEND_IP}:{SEND_UDP_PORT}")
            message = struct.pack(f'{thruster_input.size}f', *thruster_input)
            # message = struct.pack("!" + "f"*len(data), *data)
            sock_send.sendto(message, (SEND_IP, SEND_UDP_PORT))
            end = time.time()

            duration = end - start
            if (Ts - duration) < 0:
                print("Solve time of controller to high.")
                continue
            time.sleep(Ts - duration)

            print("----------------------------------------------------")

    
except KeyboardInterrupt:
    print("\nReceiver shutting down.")
finally:
    sock_receive.close()