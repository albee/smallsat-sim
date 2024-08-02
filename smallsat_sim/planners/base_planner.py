import numpy as np
import mujoco
from mujoco import mjx
from abc import abstractmethod

from smallsat_sim.envs.base_env_config import BaseEnvConfig
from smallsat_sim.envs.base_env import BaseEnv


class BasePlanner(object):
    """
    Base class of planner objects.
    """

    def __init__(self, env: BaseEnv) -> None:
        # Save some important parameters from the environment
        self.env_cfg = env.env_cfg

        # Check if there is a viewer. In case there is not,
        # dynamically allocate the visualize method to a lambda
        # function doing nothing.
        if env.viewer:
            self.viewer = env.viewer
        else:
            self.visualize = lambda *args, **kwargs: None

        # Check if there is a renderer. In case there is not,
        # dynamically allocate the visualize_renderer method
        # to a lambda function doing nothing.
        self.using_rl = env.using_rl
        if env.renderer:
            self.renderer = env.renderer
            self.frames = env.frames
            self.data = env.data
            self.cam = env.cam
            if self.using_rl:
                self.data_vec = env.data_vec
                self.model = env.model
                self.mjx_batch = env.mjx_batch

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

        # Iterate over all points which need to be visualized
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

        # Update renderer if needed
        self._visualize_renderer(points=points, color=color, size=size)

    def _visualize_renderer(
        self, points: list[np.ndarray], color=[0, 0, 1, 2], size=[0.05, 0, 0]
    ) -> None:
        """
        Visualizes reference points in the MuJoCo renderer.
        """
        if (
            self.data.time >= self.env_cfg.renderer.start_recording
            and self.data.time <= self.env_cfg.renderer.end_recording
        ):
            # Update the renderer scene
            if not self.using_rl:
                self.renderer.update_scene(self.data, self.cam)
            else:
                mjx.get_data_into(self.data_vec, self.model, self.mjx_batch)
                self.renderer.update_scene(self.data_vec[0], self.cam)

            # Iterate over all points which need to be visualized in renderer
            for point in points:
                self.renderer.scene.ngeom += 1
                x = point[0]
                y = point[1]
                z = point[2]
                mujoco.mjv_initGeom(
                    self.renderer.scene.geoms[self.renderer.scene.ngeom - 1],
                    type=mujoco.mjtGeom.mjGEOM_SPHERE,
                    size=size,
                    pos=np.array([x, y, z]),
                    mat=np.eye(3).flatten(),
                    rgba=np.array(color),
                )

            # Extract image from renderer and append it for post-processing
            sim_img = self.renderer.render().copy()
            self.frames.append(sim_img)
