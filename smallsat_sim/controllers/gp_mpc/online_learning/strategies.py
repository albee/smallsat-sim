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
from zero_order_gpmpc.models.gpytorch_models.gpytorch_utils import to_tensor, to_numpy

import numpy as np
import torch
import gpytorch
from zero_order_gpmpc.models.gpytorch_models.gpytorch_residual_model import (
    FeatureSelector,
)

from smallsat_sim.external.zero_order_gp_mpc_package.external.gpytorch_utils.gp_hyperparam_training import (
    train_gp_model,
)


class SlidingWindow(OnlineLearningStrategy):
    """
    This Dataprocessing strategy updates data by using a sliding window over
    past observations, keeping the most recent ones and discarding older data
    """

    def __init__(self, max_num_points: int = 200, device: str = "cpu") -> None:
        # Initialize superclass
        super().__init__(max_num_points, device)

        # Initialize SlidingWindow-specific variables
        self.timestamps = []

        # How many times called process
        self.counter = 0

    def process(
        self,
        gp_model: ExactGP,
        x_input: np.array,
        y_target: np.array,
        gp_feature_selector: FeatureSelector,
        timestamp: float,
    ) -> ExactGP | None:

        self.counter += 1

        # Convert to tensor
        if not torch.is_tensor(x_input):
            x_input = to_tensor(arr=x_input, device=self.device)

        if not torch.is_tensor(y_target):
            y_target = to_tensor(arr=y_target, device=self.device)

        # Extend to 2D for further computation
        x_input = torch.atleast_2d(x_input)
        y_target = torch.atleast_2d(y_target)

        if (
            gp_model.prediction_strategy is None
            or gp_model.train_inputs is None
            or gp_model.train_targets is None
        ):
            if gp_model.train_inputs is not None:
                raise RuntimeError(
                    "train_inputs in GP is not None. Something went wrong."
                )

            # Set the training data and return (in-place modification)
            gp_model.set_train_data(
                gp_feature_selector(x_input),
                y_target,
                strict=False,
            )

            # Record datapoint in timestamps
            self.timestamps.append(timestamp)

            return

        # Check if GP is already full
        if gp_model.train_inputs[0].shape[-2] >= self.max_num_points:
            with torch.no_grad():
                # Initialize selector
                selector = torch.ones(self.max_num_points, requires_grad=False)

                # Find oldest datapoint to drop
                drop_idx = self.timestamps.index(min(self.timestamps))
                selector[drop_idx] = 0

                # Calculate fantasy model with data selector
                fantasy_model = gp_model.get_fantasy_model(
                    gp_feature_selector(x_input),
                    y_target,
                    data_selector=selector,
                )

                # Update timestamps
                self.timestamps.pop(drop_idx)
                self.timestamps.append(timestamp)

            # Check for training
            fantasy_model = self._check_training(fantasy_model)

            return fantasy_model

        with torch.no_grad():
            # Add observation and return updated model
            fantasy_model = gp_model.get_fantasy_model(
                gp_feature_selector(x_input), y_target
            )

            # Record datapoint in timestamps
            self.timestamps.append(timestamp)

            return fantasy_model

    def _check_training(self, fantasy_model: ExactGP) -> ExactGP:

        if self.counter % self.max_num_points == 0 and False:
            # Before training
            for name, param in fantasy_model.named_parameters():
                print(f"Parameter {name} has shape {param.shape} and values:")
                print(param)

            fantasy_model, _ = train_gp_model(
                fantasy_model,
                torch_seed=456,
                training_iterations=300,
            )

            # After training
            for name, param in fantasy_model.named_parameters():
                print(f"Parameter {name} has shape {param.shape} and values:")
                print(param)

            return fantasy_model
        else:
            return fantasy_model
