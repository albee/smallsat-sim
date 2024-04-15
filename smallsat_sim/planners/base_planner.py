import numpy as np
import mujoco
import itertools


class BasePlanner(object):
    """
    Base class of planner objects.
    """

    def __init__(self, env) -> None:
        self.viewer = env.viewer

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns a reference point based on current observations
        """
        pass

    def visualize(self, points: list[np.ndarray]):
        """
        Visualizes reference points in the MuJoCo viewer
        """
        self.viewer.user_scn.ngeom = 0
        i = 0
        for point in points:
            x = point[0]
            y = point[1]
            z = point[2]
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[i],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=[0.05, 0, 0],
                pos=np.array([x, y, z]),
                mat=np.eye(3).flatten(),
                rgba=np.array([1, 0, 0, 2]),
            )
            i += 1
        self.viewer.user_scn.ngeom = i
