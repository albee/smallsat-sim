"""
Simple UDP Receiver. Receives data packages over and UDP socket and prints the data.
"""
import socket

# Set up the UDP receiver
UDP_IP = "127.0.0.1"  # Localhost, My current IP address
UDP_PORT = 5005       # Port to listen on

# Create a socket
sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock.bind((UDP_IP, UDP_PORT))

print(f"Listening for messages on {UDP_IP}:{UDP_PORT}...")

# Receive messages in a loop
try:
    while True:
        data, addr = sock.recvfrom(1024)  # Buffer size is 1024 bytes
        # print(f"Received message from {addr}: {data.decode()}")
except KeyboardInterrupt:
    print("\nReceiver shutting down.")
finally:
    sock.close()