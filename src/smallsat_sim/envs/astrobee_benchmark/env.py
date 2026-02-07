import mujoco

from smallsat_sim.envs.base_env import BaseEnv
from smallsat_sim.envs.perturbations import (
    PerturbationList,
    StuckOffThrusters,
    StuckOnThrusters,
    FaultyValve,
    SaturatedThrust,
    ThrustInstability,
)
from smallsat_sim.utils import xml_parser_lightweight


class AstrobeeBenchmarkEnv(BaseEnv):
    """
    MuJoCo environment for benchmarking classic controllers against the RL scene.
    Uses the lightweight XML (no ISS), with an optional floor for visualization.
    """

    def __init__(self, args) -> None:
        # Load classic controller config but render the lightweight scene.
        self.env_cfg, self.model_cfg = self._load_cfg(
            env_name="astrobee", model_name="astrobee"
        )
        self.env_cfg.sim.include_floor = True
        super().__init__(args=args)

        verbose = getattr(self.env_cfg.sim, "verbose", False)
        self.perturbations = PerturbationList(
            [
                StuckOffThrusters(self.model_cfg, verbose=verbose),  # 0
                StuckOnThrusters(self.model_cfg, verbose=verbose),  # 1
                FaultyValve(self.model_cfg, verbose=verbose),  # 2
                SaturatedThrust(self.model_cfg, verbose=verbose),  # 3
                ThrustInstability(self.model_cfg, verbose=verbose),  # 4
            ]
        )

    def _setup_sim(self, args):
        """
        Prepares simulation according to args using the lightweight XML parser.
        """
        # Generate xml using env and model config files
        xml = xml_parser_lightweight.generate_mujoco_xml(self.env_cfg, self.model_cfg)
        # Create model and data instances
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)

        # Launch the viewer
        if not args.headless:
            self._create_viewer()
        else:
            self.viewer = None
            self._update_viewer = lambda *args, **kwargs: None

        # Launch the renderer to create a video
        if args.video:
            self._create_renderer()
        else:
            self.renderer = None
            self._update_renderer = lambda *args, **kwargs: None
