# from zero_order_gpmpc.models.gpytorch_models import FeatureSelector
from l4acados.models.pytorch_models.pytorch_feature_selector import PyTorchFeatureSelector
import torch
import gpytorch
import numpy as np


# class ScaleFeatureSelector(FeatureSelector):
class ScaleFeatureSelector(PyTorchFeatureSelector):
    def __init__(
        self,
        input_selection: np.ndarray | torch.Tensor | None = None,
        external_inputs: np.ndarray | torch.Tensor | None = None,
        scale=1,
        device="cpu",
    ) -> torch.Tensor:
        input_selection = input_selection * scale
        super().__init__(input_selection, external_inputs, device)


class ResidualScaler:
    def __init__(self, scale=100) -> None:
        self.scale = scale
        self.scale_inv = 1 / scale

    def __call__(
        self,
        x_input: (
            torch.Tensor | gpytorch.distributions.MultitaskMultivariateNormal
        ) = None,
    ):

        if isinstance(x_input, torch.Tensor):
            return x_input * self.scale
        elif isinstance(x_input, gpytorch.distributions.MultitaskMultivariateNormal):
            x_input.mean.div_(self.scale)
            x_input.covariance_matrix.div_(self.scale ** 2)

            return x_input

        else:
            raise ValueError("Unsupported input type")
