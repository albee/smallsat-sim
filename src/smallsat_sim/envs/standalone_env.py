import numpy as np
import jax.numpy as jnp

class StandaloneEnv(object):
    def __init__(self, env_cfg, model) -> None:
        self._obs = None
        self.env_cfg = env_cfg
        self.symbolic_model = model

        # This is required from planner. Set this to none here
        self.viewer = None
        self.renderer = None

        self.logging_enabled = False

    def reset(self) -> None:
        """
        Resets environment to a desired state.
        """
        pass

    def set_obs(self, qpos: np.ndarray, v: np.ndarray, qvel: np.ndarray, v_frame: str = "inertial") -> np.ndarray:
        """
        Return all states

        args:
            v_frame (str): Specifies the frame of the velocity in the returned observation.
                             - "body": Return the velocity in the body frame.
                             - "inertial": Return the velocity in the inertial frame.

        returns:
            np.array: Array of observations containing position, orientation, velocity
                      and angular velocity.
        """
        # obs = [r (3),
        #        q (4),
        #        v (3), --> in body/inertial frame
        #        omega (3)]


        if v_frame == "body":
            raise NotImplementedError(
                f"Velocity refernce body is not implented"
            )
        
        elif v_frame == "inertial":
            pass # Nothing to do, velocity is already in inertial frame.

        else:
            raise RuntimeError(
                f"Specified velocity frame {v_frame} not valid. "
                "Must be either 'body' or 'inertial'."
            )

        # Create array of observations
        self._obs = np.concatenate((qpos, v, qvel))

    @property
    def obs(self):
        return self.get_obs()
    
    def get_obs(self) -> np.ndarray | jnp.ndarray:
        """
        Returns the current observations
        """
        if self._obs is None:
            raise RuntimeError(
                "Observation not initialized. Did you call  set_obs()?"
            )
        return self._obs.copy()
