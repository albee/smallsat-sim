import socket
import time
import json

# Server Configuration
HOST = "0.0.0.0"  # Listen on all interfaces
PORT = 1153  # Same as Vicon Tracker default port

# Fake Object Data
vicon_data = {
    "MyObject": {
        "position": [1.0, 2.0, 3.0],  # Fake X, Y, Z
        "orientation": [0.0, 0.0, 0.0, 1.0]  # Fake Quaternion (x, y, z, w)
    }
}

# Start UDP Server
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
#sock.bind((HOST, PORT))

print(f"Simulated Vicon Server running on {HOST}:{PORT}...")

while True:
    # Simulate Vicon system by sending fake data every 0.1s
    message = json.dumps(vicon_data)
    sock.sendto(message.encode(), ("127.0.0.1", 1153))  # Send to client (modify IP as needed)
    time.sleep(0.1)
