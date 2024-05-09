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


def cost(p1: np.ndarray, p2: np.ndarray, directions: dict) -> float:
    """
    Returns the cost of moving from point p1 to point p2.
    """
    collide, dist = check_collision(p1, p2)
    if collide:
        return np.inf
    return get_distance(p1, p2)


def cost(self, u, v):
    """
    Calculate the cost of moving from node u to node v based on predefined directions.
    """
    direction = tuple(np.array(v) - np.array(u))
    return self.directions.get(direction, np.inf)  # Return inf if direction is not defined


def check_collision(p1: np.ndarray, p2: np.ndarray) -> tuple[bool, float]:
    """
    [Placeholder] Returns True if there is a collision between points p1 and p2, False otherwise.
    """
    # TODO: Implement MuJoCo ray casting to check for collisions with meshes
    # (mj_ray, mj_rayMesh?)
    return False, 0.0  # Placeholder return value


def heuristic(p: np.ndarray, goal: np.ndarray) -> float:
    """
    Returns the heuristic (Euclidean distance) cost from point p to the goal
    """
    return get_distance(p, goal)