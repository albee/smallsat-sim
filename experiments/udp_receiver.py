import socket
import struct

UDP_IP = "127.0.0.1"  # Pc IP
# UDP_IP = "0.0.0.0"  # Listen on all interfaces
UDP_PORT = 1153

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for control signals on {UDP_IP}:{UDP_PORT}...")
EXPECTED_LENGTH = 12  # Expecting 12 floats

while True:
    data, addr = sock.recvfrom(4 * EXPECTED_LENGTH)  # 4 bytes per float * 12
    data_array = struct.unpack(f'{EXPECTED_LENGTH}f', data)  # Unpack 12 floats
    # control_signal = struct.unpack('f', data)[0]
    print("Received Data:", data_array)
    
    # print(f"Received Control Signal: {control_signal}")
