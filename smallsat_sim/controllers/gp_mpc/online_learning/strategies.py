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


class SlidingWindow(DataProcessingStrategy):
    """
    Implements the sliding window approach to adapt the points
    used for posterior predictions
    """

    raise NotImplementedError(
        "The SlidingWindow approach has not been implemented yet."
    )
