import numpy as np
import mujoco
from mujoco import mjx
import itertools

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
        self.using_rl = env.using_rl
        if self.args.video:
            self.frames = env.frames
            self.data = env.data
            self.cam = env.cam
            if self.using_rl:
                self.data_vec = env.data_vec
                self.model = env.model
                self.mjx_batch = env.mjx_batch

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        """
        Returns a reference point based on current observations
        """
        pass

    def visualize(self, points: list[np.ndarray], color=[1, 0, 0, 2]):
        """
        Visualizes reference points in the MuJoCo viewer
        """
        self.viewer.user_scn.ngeom = 0
        i = 0
        if (
            self.args.video
            and self.data.time >= BaseEnvConfig.renderer.start_recording
            and BaseEnvConfig.renderer.end_recording
        ):  # TODO: change to env config
            if not self.using_rl:
                self.renderer.update_scene(self.data, self.cam)
            else:
                mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)
                self.renderer.update_scene(self.data_vec[0], self.cam)
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
                rgba=np.array(color),
            )
            if (
                self.args.video
                and self.data.time >= BaseEnvConfig.renderer.start_recording
                and self.data.time <= BaseEnvConfig.renderer.end_recording
            ):
                self.renderer.scene.ngeom += 1
                mujoco.mjv_initGeom(
                    self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=[0.05, 0, 0],
                    pos=np.array([x, y, z]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array([1, 0, 0, 2]),
                )
            i += 1
        if (
            self.args.video
            and self.data.time >= BaseEnvConfig.renderer.start_recording
            and BaseEnvConfig.renderer.end_recording
        ):
            sim_img = self.renderer.render().copy()
            self.frames.append(sim_img)
        self.viewer.user_scn.ngeom = i
