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
from typing import Optional
from zero_order_gpmpc.models.gpytorch_models.gpytorch_residual_model import (
    FeatureSelector,
)

from smallsat_sim.external.zero_order_gp_mpc_package.external.gpytorch_utils.gp_hyperparam_training import (
    train_gp_model,
)

from smallsat_sim.controllers.gp_mpc.online_learning.utils import ResidualScaler
from scipy.stats import norm


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
        self.counter = -1

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

            # print(fantasy_model.covar_module.base_kernel.variance.data)
            # print(fantasy_model.likelihood.noise.data)

            return fantasy_model
        else:
            return fantasy_model


class SlidingWindowPlus(OnlineLearningStrategy):
    def __init__(self, max_num_points: int = 200, device: str = "cpu") -> None:
        # Initialize superclass
        super().__init__(max_num_points, device)

        # Initialize SlidingWindow-specific variables
        self.timestamps = []

        # How many times called process
        self.counter = -1

        # Initialize fault detector flag counter
        self.fault_counter = 0

        # Initialize past input
        self.x_input_past = torch.zeros((1,27))

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
            print(timestamp)
            

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

            # Save point
            self.x_input_past = x_input.clone()

            return
        print(gp_model.train_inputs[0].shape[-2])

        # Check if point is different enough to last one

        # Check if the point shall be added to the dictionary
        if self._detect_failure(
            gp_model, x_input, y_target, gp_feature_selector, timestamp
        ):
            return gp_model
        
        if True:
            delta = torch.norm(gp_feature_selector(x_input-self.x_input_past), 2)
            if delta < 0.01:
                print(f"Rejected point due to low delta of {delta}")
                return gp_model

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

            # Save point
            self.x_input_past = x_input.clone()

            return fantasy_model

        with torch.no_grad():
            # Add observation and return updated model
            fantasy_model = gp_model.get_fantasy_model(
                gp_feature_selector(x_input), y_target
            )

            # Record datapoint in timestamps
            self.timestamps.append(timestamp)

            # Save point
            self.x_input_past = x_input.clone()

            return fantasy_model

    def _check_training(self, fantasy_model: ExactGP) -> ExactGP:
        return fantasy_model

    def _detect_failure(
        self,
        gp_model: gpytorch.models.ExactGP,
        x_test: torch.Tensor,
        y_test: torch.Tensor,
        gp_feature_selector: FeatureSelector,
        timestamp: float,
    ) -> bool:
        """
        Checks all criteria if point shall be added to dictionary
        """
        # Calculate the expected mean at the location
        with torch.no_grad():
            z_test = torch.atleast_2d(gp_feature_selector(x_test))
            observed_pred = gp_model(z_test)

            # Extract mean
            mu = observed_pred.mean
            # print(mu)

            # Extract Variance
            stddev = observed_pred.stddev

        # Calculate the beta quantity
        beta = self._calc_beta(mu, y_test, stddev)

        if np.min(beta) < 0.05 and gp_model.train_inputs[0].shape[0] > 50:
            print(f"Fault at {timestamp} due to idx {np.argmin(beta)} with value{np.min(beta)}!")
            self.flag_counter += 1
            if self.flag_counter == 5:
                gp_model.set_train_data(
                    gp_feature_selector(x_test),
                    y_test,
                    strict=False,
                )
                print(f"Reset GP data at time {timestamp}!")

                return True
            else:
                return False
        else:
            self.flag_counter = 0

            return False

    def _calc_beta(self, mu: torch.Tensor, y_test: torch.Tensor, stddev: torch.Tensor):

        # Calculate the standardized distance for each dimension
        standardized_distance = torch.abs(mu - y_test) / stddev

        # # Calculate the argument for the normal CDF for each dimension
        # cdf_args = 1.96 - standardized_distance

        # # Calculate the probability using the CDF of the standard normal distribution
        # beta = norm.cdf(cdf_args.cpu().numpy())

        p = 2 * (1-norm.cdf(standardized_distance))

        return p
