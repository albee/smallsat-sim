import numpy as np
import socket
import json
import threading
import time

# Initialize a global list to cache messages
message_cache = []

class UdpBuffer:
    """
    A class for managing UDP communication with a shared data buffer.

    Attributes:
    ----------
    _UDP_IP : str
        The IP address to listen for UDP messages.
    _UDP_PORT : int
        The port to listen for UDP messages.
    _socket : socket.socket
        A socket object for UDP communication.
    _lock : threading.Lock
        A lock to ensure thread-safe access to the shared data cache.
    _data : str or None
        The most recently received data.

    Methods:
    -------
    _process():
        Continuously listens for incoming UDP messages and updates the shared data cache.
    start():
        Starts listening for incoming messages in a background thread.
    stop():
        Stops the background thread (not implemented).
    data():
        Retrieves the most recent data (not implemented).
    """

    def __init__(self, UDP_IP="127.0.0.1", UDP_PORT=5006):
        """
        Initializes the UdpBuffer object.

        Parameters:
        ----------
        UDP_IP : str, optional
            The IP address to listen for UDP messages (default is "127.0.0.1").
        UDP_PORT : int, optional
            The port to listen for UDP messages (default is 5006).
        """
        self._UDP_IP = UDP_IP
        self._UDP_PORT = UDP_PORT
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((UDP_IP, UDP_PORT))
        self._lock = threading.Lock()  # A lock to manage access to the shared cache
        self._data = None
        self._running = False
        print(f"Receiving messages from {UDP_IP}:{UDP_PORT}. Press Ctrl+C to stop.")
    
    def _process(self):
        """
        Listens for incoming UDP messages in an infinite loop and updates the data cache.
        """
        while self._running:
            try:
                # Receive data from the socket
                data, addr = self._socket.recvfrom(1024)  # Buffer size is 1024 bytes
                # Safely append the message to the cache
                with self._lock:
                    self._data = data
                time.sleep(0.001)
            except Exception as e:
                print(f"Error receiving data: {e}")
                break

    def start(self):
        """
        Starts listening for incoming UDP messages in a background thread.
        """
        self._running = True
        self._receiver_thread = threading.Thread(target=self._process, daemon=True)
        self._receiver_thread.start()

    def stop(self):
        """
        Stops the background thread for listening to UDP messages.
        """
        self._running = False

    @property
    def data(self):
        """
        Retrieves the most recent data received.
        """
        with self._lock:
            return self._data


class UdpReceiver:
    """
    A class for receiving control input via UDP.

    Attributes:
    ----------
    _UDP_IP : str
        The IP address to listen for UDP messages.
    _UDP_PORT : int
        The port to listen for UDP messages.
    _socket : socket.socket
        A socket object for UDP communication.

    Methods:
    -------
    get_control_input() -> np.ndarray:
        Receives and parses control input from the UDP socket.
    """

    def __init__(self, UDP_IP="172.17.0.2", UDP_PORT=5011):
        """
        Initializes the UdpReceiver object.

        Parameters:
        ----------
        UDP_IP : str, optional
            The IP address to listen for UDP messages (default is "127.0.0.1").
        UDP_PORT : int, optional
            The port to listen for UDP messages (default is 5006).
        """
        self._UDP_IP = UDP_IP
        self._UDP_PORT = UDP_PORT
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._socket.bind((UDP_IP, UDP_PORT))
        print(f"Receiving messages from {UDP_IP}:{UDP_PORT}. Press Ctrl+C to stop.")

    def get_control_input(self) -> np.ndarray:
        """
        Receives control input data via UDP and parses it into a NumPy array.

        Returns:
        -------
        np.ndarray
            The control input as a NumPy array.
        """
        ctrl_input_str, addr = self._socket.recvfrom(1024)  # Buffer size is 1024 bytes
        return np.array(json.loads(ctrl_input_str.decode()))


class UdpPublisher:
    """
    A class for publishing simulation states via UDP.

    Attributes:
    ----------
    _UDP_IP : str
        The IP address to send UDP messages to.
    _UDP_PORT : int
        The port to send UDP messages to.
    _socket : socket.socket
        A socket object for UDP communication.

    Methods:
    -------
    publish_state(r, q, v, omega):
        Sends the simulation state data as a UDP message.
    """

    def __init__(self, UDP_IP="127.0.0.1", UDP_PORT=5011):
        """
        Initializes the UdpPublisher object.

        Parameters:
        ----------
        UDP_IP : str, optional
            The IP address to send UDP messages to (default is "127.0.0.1").
        UDP_PORT : int, optional
            The port to send UDP messages to (default is 5005).
        """
        self._UDP_IP = UDP_IP
        self._UDP_PORT = UDP_PORT
        self._socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        print(f"Sending messages to {UDP_IP}:{UDP_PORT}. Press Ctrl+C to stop.")

    def publish_state(self, r, q, v, omega):
        """
        Sends the simulation state data as a JSON-formatted UDP message.

        Parameters:
        ----------
        r : np.ndarray
            Position vector.
        q : np.ndarray
            Orientation quaternion.
        v : np.ndarray
            Linear velocity vector.
        omega : np.ndarray
            Angular velocity vector.
        """
        data = {
            "r": r.tolist(),
            "q": q.tolist(),
            "v": v.tolist(),
            "omega": omega.tolist()
        }
        data_str = json.dumps(data)
        self._socket.sendto(data_str.encode(), (self._UDP_IP, self._UDP_PORT))


def create_communication_channels():
    """
    Creates and returns a UdpReceiver and UdpPublisher instance.

    Returns:
    -------
    tuple(UdpReceiver, UdpPublisher)
        The receiver and publisher objects.
    """
    receiver = UdpReceiver()
    publisher = UdpPublisher()
    return receiver, publisher

