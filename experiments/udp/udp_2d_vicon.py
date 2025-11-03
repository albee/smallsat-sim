# import ViconDataStream
from pyvicon_datastream import tools
import socket
import time
import math
import numpy as np
import sys

###### HELPER METHODS ######
def get_3d_dist_between_points(pos1, pos2):
    pos1_x, pos1_y, pos1_z = pos1[0], pos1[1], pos1[2]
    pos2_x, pos2_y, pos2_z = pos2[0], pos2[1], pos2[2]

    x = (pos1_x - pos2_x)**2
    y = (pos1_y - pos2_y)**2
    z = (pos1_z - pos2_z)**2
    dist_3D = math.sqrt(x + y + z)

    return dist_3D

def get_xyz_position(tracker_inst, obj_name):
    latency, frameno, position = tracker_inst.get_position(obj_name)
    if position != []:
        xyz_position = position[0][2:5] # get x,y,z only
        return xyz_position
    return []

def get_xyz_position_orientation(tracker_inst, obj_name):
    latency, frameno, position = tracker_inst.get_position(obj_name)
    if position != []:
        xyz_position = position[0][2:5] # get x,y,z only
        orientation = position[0][7] # get rotation around z axis
        return xyz_position, orientation
    return []

def get_xyz_position_orientation_quat(tracker_inst, obj_name):
    latency, frameno, position = tracker_inst.get_position(obj_name)
    if position != []:
        xyz_position = position[0][2:5] # get x,y,z only
        orientation = position[0][7] # get rotation around z axis
        orientation_quat = [math.cos(orientation / 2), 0, 0, math.sin(orientation /2)]
        return xyz_position, orientation_quat
    return []

def get_velocity(tracker_inst, obj_name):
    # TODO Implement
    return [0, 0, 0]

def get_angular_vel(tracker_inst, obj_name):
    # TODO Implement
    return [0]

def get_static_position(tracker_inst, obj_name, collect_frame_no):
    valid_positions = []
    frames, skips = 0, 0
    while frames < collect_frame_no:
        xyz_position = get_xyz_position(tracker_inst, obj_name)
        if xyz_position != []:
            valid_positions.append(xyz_position)
            frames += 1
        else:
            skips += 1

        if skips > collect_frame_no:
            print(f"Too many Frames Skipped (>{skips})")
            return None

    # Calculate Median
    valid_positions = np.asarray(valid_positions)
    return  np.median(valid_positions, axis=0)

def round_list(list, digits):
    return [ round(elem, digits) for elem in list ]
############################

# Vicon server IP and VOXL2 details
VICON_SERVER_IP = "127.0.0.1"  # Replace with your Vicon server IP
VOXL2_IP = "127.0.0.1"  # Replace with your VOXL2 IP
VOXL2_PORT = 50011  # UDP port for VOXL2

# Intialize
print("Connecting to Vicon Tracker…")
vicontracker = tools.ObjectTracker(VICON_SERVER_IP)
# Check available objects
print("Tracked Objects:", vicontracker.get_object_names())
# Get pose of a specific object
OBJECT_NAME = "Smallsat"

# Get initial start position by collecting a number of frames and calculate median
static_start_position = get_static_position(vicontracker, OBJECT_NAME, 50)
if static_start_position is not None:
    print(f"Static Start Position ({OBJECT_NAME}) XYZ: {round_list(static_start_position, 0)}")
else:
    print("Could not determine Static Start Position")
    sys.exit()

# Create UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

while True:
    xyz_position, orientation_quat = get_xyz_position_orientation_quat(vicontracker, OBJECT_NAME)
    velocity = get_velocity(vicontracker, OBJECT_NAME)
    angular_velocity = get_angular_vel(vicontracker, OBJECT_NAME)

    # Format data as a string
    tx = xyz_position[0]
    ty = xyz_position[1]
    tz = xyz_position[2]
    qw = orientation_quat[0]
    qx = orientation_quat[1]
    qy = orientation_quat[2]
    qz = orientation_quat[3]
    vx = velocity[0]
    vy = velocity[1]
    vz = velocity[2]
    yaw_rate = angular_velocity[0]

    pose_velocity_data = f"{tx},{ty},{tz},{qw, qx, qy, qz},{vx},{vy},{vz},{yaw_rate}"

    # Send data to VOXL2 via UDP
    sock.sendto(pose_velocity_data.encode(), (VOXL2_IP, VOXL2_PORT))
    
    time.sleep(0.01)  # Adjust for desired frequency