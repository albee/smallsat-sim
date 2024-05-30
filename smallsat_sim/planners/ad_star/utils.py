import numpy as np


def get_distance(p1: tuple | np.ndarray, p2: tuple | np.ndarray) -> float:
    """
    Returns the Euclidean distance between two points p1 and p2.
    """
    p1 = np.array(p1)
    p2 = np.array(p2)

    return np.linalg.norm(p1 - p2)


def is_in_bound(p: np.ndarray, bounds: np.ndarray, factor: float = 0.0) -> bool:
    """
    Returns True if point p is within bounds, False otherwise.
    """
    return np.all(p >= bounds[0] + factor) and np.all(p <= bounds[1] - factor)


def heuristic(p: np.ndarray, goal: np.ndarray) -> float:
    """
    Returns the heuristic (Euclidean distance) cost from point p to the goal
    """
    return get_distance(p, goal)