from gpytorch.models.exact_gp import ExactGP
from numpy.core.multiarray import array as array
from zero_order_gpmpc.models.gpytorch_models.gpytorch_residual_learning_model import (
    DataProcessingStrategy,
    ResidualGaussianProcess,
    VoidDataStrategy,
    RecordDataStrategy,
    OnlineLearningStrategy,
    GPyTorchResidualLearningModel,
)

import numpy as np
import torch
from zero_order_gpmpc.models.gpytorch_models.gpytorch_residual_model import (
    FeatureSelector,
)


class SlidingWindow(OnlineLearningStrategy):
    """
    This Dataprocessing strategy updates data by using a sliding window over
    past observations, keeping the most recent ones and discarding older data
    """

    def __init__(self, max_num_points: int = 200, device: str = "cpu") -> None:
        super().__init__(max_num_points, device)

    def process(
        self,
        gp_model: ExactGP,
        x_input: np.array,
        y_target: np.array,
        gp_feature_selector: FeatureSelector,
    ) -> ExactGP | None:
        pass
