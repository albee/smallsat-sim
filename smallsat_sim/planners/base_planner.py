import numpy as np
import mujoco
from abc import abstractmethod

from smallsat_sim.envs.base_env_config import BaseEnvConfig


class BasePlanner(object):
    """
    Base class of planner objects.
    """

    def __init__(self, env) -> None:
        # Check if there is a viewer. In case there is not,
        # dynamically allocate the visualize method to a lambda
        # function doing nothing.
        if env.viewer:
            self.viewer = env.viewer
        else:
            self.visualize = lambda *args, **kwargs: None

        # Use parser args
        self.args = env.args

        # Get the renderer to visulaize the reference
        self.renderer = env.renderer
        if self.args.video:
            self.frames = env.frames
            self.data = env.data
            self.cam = env.cam

    @abstractmethod
    def get_reference(self, obs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Returns a reference for the position and attitude based on current observations
        """
        pass

    def visualize(
        self, points: list[np.ndarray], color=[1, 0, 0, 2], size=[0.05, 0, 0]
    ):
        """
        Visualizes reference points in the MuJoCo viewer
        """
        self.viewer.user_scn.ngeom = 0

        i = 0
        if self.args.video and self.data.time >= BaseEnvConfig.renderer.start_recording and BaseEnvConfig.renderer.end_recording:
            self.renderer.update_scene(self.data, self.cam)
        for point in points:
            self.viewer.user_scn.ngeom += 1
            x = point[0]
            y = point[1]
            z = point[2]
            mujoco.mjv_initGeom(
                self.viewer.user_scn.geoms[self.viewer.user_scn.ngeom - 1],
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=size,
                pos=np.array([x, y, z]),
                mat=np.eye(3).flatten(),
                rgba=np.array(color),
            )

            if self.args.video and self.data.time >= BaseEnvConfig.renderer.start_recording and self.data.time <= BaseEnvConfig.renderer.end_recording:
                self.renderer.scene.ngeom += 1
                mujoco.mjv_initGeom(
                    self.renderer.scene.geoms[self.renderer.scene.ngeom-1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.05, 0, 0],
                    pos=np.array([x, y, z]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1, 0, 0, 2]),
                )
            i += 1
        if self.args.video and self.data.time >= BaseEnvConfig.renderer.start_recording and BaseEnvConfig.renderer.end_recording:
            sim_img = self.renderer.render().copy()
            self.frames.append(sim_img)
        self.viewer.user_scn.ngeom = i
