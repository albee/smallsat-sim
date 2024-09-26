"""
Simple UDP Publisher. Publishes data packages over and UDP socket and prints the data.
"""
import socket
import time
import pandas as pd

# Load CSV data
df = pd.read_csv("logs/sim/2025-02-27/08:56:53 PST-0800.csv")

UDP_PORT = 1153

# Create a UDP socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

# Set the target IP and port (make sure the receivers IP is set to the same IP)
IP = "127.0.0.1"  # VOXL's IP Hotspot

# Local testing with PC (not VOXL) 
# IP = "255.255.255.255"
sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)  


# Define the data array (example: x, y, z, qw, qx, qy, qz, vx, vy, vz, yaw_rate)
# ViconState: position=(0.0, 0.0, 0.0), velocity=(0.0, 0.0, 0.0), quaternion=(0.0, 0.0, 0.0, 0.0), euler_angles=(0.0, 0.0, 0.0), angular_velocity=(0.0, 0.0, 0.0)
# data_array = [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]

# Exstract Vicon data
pos_x, pos_y, pos_z = df['obs_pos_x'], df['obs_pos_y'], df['obs_pos_z']
vel_x, vel_y, vel_z = df['obs_vel_x'], df['obs_vel_y'], df['obs_vel_z']
roll, pitch, yaw = df['obs_euler_roll'], df['obs_euler_pitch'], df['obs_euler_yaw']
quat_w, quat_x, quat_y, quat_z = df['obs_quat_w'], df['obs_quat_x'], df['obs_quat_y'], df['obs_quat_z']
roll_rot, pitch_rot, yaw_rot = df['obs_rot_roll'], df['obs_rot_pitch'], df['obs_rot_yaw']

data_length = len(df['obs_pos_x'])
# Send each row as a UDP packet
for row in range(data_length):
    data_array = [pos_x[row], pos_y[row], pos_z[row], vel_x[row], vel_y[row], vel_z[row], quat_x[row], quat_y[row], quat_z[row], quat_w[row], yaw[row], roll[row], pitch[row], yaw_rot[row], roll_rot[row], pitch_rot[row]]
    # message = f"{row['timestamp']},{row['value']}"  # Format data as CSV string
    # Convert array to a comma-separated string
    message = ",".join(map(str, data_array))
    sock.sendto(message.encode(), (IP, UDP_PORT))
    print(f"Sending: {message} to {IP}:{UDP_PORT}")
    time.sleep(0.1)  # Simulate real-time transmission

print('Sent all data.')

# Convert array to a comma-separated string
message = ",".join(map(str, data_array))

# Send the message
# sock.sendto(message.encode(), (VOXL_IP, UDP_PORT))
# Send messages in a loop
# try:
#     while True:
#         sock.sendto(message.encode(), (IP, UDP_PORT))
#         print(f"Sending: {message} to {IP}:{UDP_PORT}")
#         time.sleep(0.1)
# except KeyboardInterrupt:
#     print("\nPublsiher shutting down.")
# finally:
#     sock.close()

