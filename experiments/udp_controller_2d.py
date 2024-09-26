import numpy as np

from smallsat_sim.envs.dynamics_2d import SymbolicModel2D
from smallsat_sim.controllers.mpcc.controller_2d import NominalMPCCController2D
from smallsat_sim.envs.astrobee_2d.cfg.config import EnvConfig as AstroBeeEnvCfg
from smallsat_sim.model.astrobee_2d.cfg.config import ModelConfig as AstroBeeModelCfg
from smallsat_sim.envs.standalone_env import StandaloneEnv
from smallsat_sim.planners.mission.mission_2d import MissionPlanner2D
import socket
import json
import time
from smallsat_sim.communication.udp import UdpBuffer
import argparse

parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--ip', help='UDP IP', default="127.0.0.1") # IP of docker in pc
parser.add_argument('--port', help='UDP PORT', default=5011, type=int)
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
print(f"IP: {args.ip}, PORT: {args.port}.")
udp_read_buffer = UdpBuffer(args.ip, args.port)
udp_read_buffer.start()
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Listening for messages on {args.ip}:{args.port}...")
print(f"Sending messages to {args.ip}:{args.port+1}...")

# Receive messages in a loop
try:
    while True:
        start = time.time()
        data = udp_read_buffer.data
        if data is None:
            print("No data received. Waiting.")
            time.sleep(0.25)
            continue
        data_str = data.decode()
        # print(f"Received message: {data.decode()}")
        data = json.loads(data_str)
        r, q, v, omega = np.array(data["r"]),np.array(data["q"]),np.array(data["v"]), np.array(data["omega"])
        # print("r: ", r, " q: ", q, " v: ", v, " omega: ", omega)
        # Set observations received in enviroment for controller
        env.set_obs(np.concatenate([r,q]), v, omega)

        if ctrl is None:
            ctrl = NominalMPCCController2D(env, planner)

        control_input = ctrl.get_control_input(env)

        ctrl_input_str = json.dumps(control_input.tolist())
        sock_send.sendto(ctrl_input_str.encode(), (args.ip,args.port+1))
        end = time.time()

        duration = end - start
        if (Ts - duration) < 0:
            print("Solve time of controller to high.")
            continue
        time.sleep(Ts - duration)

except KeyboardInterrupt:
    print("\nReceiver shutting down.")
finally:
    sock_send.close()