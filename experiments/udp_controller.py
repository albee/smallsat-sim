import numpy as np

from smallsat_sim.envs.dynamics import SymbolicModel
from smallsat_sim.controllers.mpcc.controller import NominalMPCCController
from smallsat_sim.envs.astrobee.cfg.config import EnvConfig as AstroBeeEnvCfg
from smallsat_sim.model.astrobee.cfg.config import ModelConfig as AstroBeeModelCfg
from smallsat_sim.envs.standalone_env import StandaloneEnv
from smallsat_sim.planners.mission.mission import MissionPlanner
import socket
import json

import argparse
parser = argparse.ArgumentParser(description='Description of your program')
parser.add_argument('--ip', help='UDP IP', default="127.0.0.1")
parser.add_argument('--port', help='UDP PORT', default=5005, type=int)
args = parser.parse_args()

cfg = AstroBeeModelCfg()
model = SymbolicModel(cfg)
env = StandaloneEnv(env_cfg=AstroBeeEnvCfg(), model=model)
env.model_cfg = cfg
planner =  MissionPlanner(env) 


env.set_obs(np.array([0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]), np.zeros(3), np.zeros(3))

ctrl = None

# Create a socket
sock_read = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
sock_read.bind((args.ip, args.port))
sock_send = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

print(f"Listening for messages on {args.ip}:{args.port}...")
print(f"Sending messages to {args.ip}:{args.port+1}...")

# Receive messages in a loop
try:
    while True:
        data, addr = sock_read.recvfrom(1024)  # Buffer size is 1024 bytes
        data_str = data.decode()
        # print(f"Received message from {addr}: {data.decode()}")
        data = json.loads(data_str)
        r, q, v, omega = np.array(data["r"]),np.array(data["q"]),np.array(data["v"]), np.array(data["omega"])
        # print("r: ", r, " q: ", q, " v: ", v, " omega: ", omega)
        # Set observations received in enviroment for controller
        env.set_obs(np.concatenate([r,q]), v, omega)

        if ctrl is None:
            ctrl = NominalMPCCController(env, planner)
        control_input = ctrl.get_control_input(env)

        ctrl_input_str = json.dumps(control_input.tolist())
        sock_send.sendto(ctrl_input_str.encode(), (args.ip,args.port+1))

except KeyboardInterrupt:
    print("\nReceiver shutting down.")
finally:
    sock_read.close()
    sock_send.close()