import numpy as np
from smallsat_sim.envs.standalone_env import StandaloneEnv

class DummyReceiver:
    def __init__(self, env: StandaloneEnv, controller):
        self._env = env
        self._controller = controller

    def get_control_input(self) -> np.ndarray:
        return self._controller.get_control_input(self._env)

class DummyPublisher:
    def __init__(self, env: StandaloneEnv):
        self._env = env

    def publish_state(self, r, q, v, omega):
        self._env.set_obs(np.concatenate([r,q]), v, omega)

def create_communication_channels(env_cfg, model, controller):
    env = StandaloneEnv(env_cfg, model)
    receiver = DummyReceiver(env, controller)
    publisher = DummyPublisher(env)
    return receiver, publisher