import numpy as np


class Perturbation(object):
    """
    Base class for all perturbations applied to the model.
    Perturbations are defined as an changes to the model's
    dynamics such as a mismatch between desired thrust versus
    actual thrust.
    """

    def __init__(self) -> None:
        pass

    def apply(self, input: np.ndarray) -> np.ndarray:
        pass


class PerturbationList(object):
    """
    Applies multiple perturbations
    """

    def __init__(self, perturbations: list[Perturbation]) -> None:
        super().__init__()
        self.perturbations = perturbations

    def apply(self, input: np.ndarray) -> np.ndarray:
        for perturbation in self.perturbations:
            input = perturbation.apply()

        return input
