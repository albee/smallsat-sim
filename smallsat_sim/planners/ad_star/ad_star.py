import numpy as np
from smallsat_sim.planners.ad_star import utils
import heapq
import time


class ADStarPlanner():
    """
    This class implements the Anytime Dynamic A* (AD*) path planning algorithm.
    It is based on:
    Maxim Likhachev, David Ferguson, Geoff Gordon, Anthony Stentz, and Sebastian Thrun. 
    “Anytime Dynamic A*: An Anytime, Replanning Algorithm.” In Proc. Int. Conf. Automated Planning and Scheduling, 15. 
    https://aaai.org/papers/icaps-05-027-anytime-dynamic-a-an-anytime-replanning-algorithm/.
    """
    def __init__(self, start: tuple, goal: tuple) -> None:  # Not using env in planner yet
        self.start = start  # Initial position of the agent
        self.goal = goal  # Goal position of the agent
        # Inconsistent states "s": Overconsistent: g(s) > rhs(s), Underconsistent: g(s) < rhs(s)
        self.OPEN = []  # Priority queue of inconsistent states to be expanded
        self.g = {}  # Dictionary of costs from each state to goal
        self.rhs = {self.goal: 0}  # Dictionary of one-step lookahead costs
        self.epsilon = 1.0  # Scaling factor for the heuristic (inflation factor)
        self.bounds = np.array([[0, 0, 0], [10, 10, 10]])  # Hardcoded bounds for the environment

        # Insert the goal into the priority queue to start the expansion
        heapq.heappush(self.OPEN, (self.key(self.goal), self.goal))
        self.CLOSED = set()  # Set of states that have been expanded
        self.INCONS = set()  # Set of states that have been expanded and are inconsistent
        #self.removed = set()  # Set of states that have been removed from the priority queue

        # Define the possible directions and their costs
        self.directions = {(1, 0, 0): 1, (0, 1, 0): 1, (0, 0, 1): 1, \
                           (-1, 0, 0): 1, (0, -1, 0): 1, (0, 0, -1): 1, \
                           (1, 1, 0): np.sqrt(2), (1, 0, 1): np.sqrt(2), (0, 1, 1): np.sqrt(2), \
                           (-1, -1, 0): np.sqrt(2), (-1, 0, -1): np.sqrt(2), (0, -1, -1): np.sqrt(2), \
                           (1, -1, 0): np.sqrt(2), (-1, 1, 0): np.sqrt(2), (1, 0, -1): np.sqrt(2), \
                           (-1, 0, 1): np.sqrt(2), (0, 1, -1): np.sqrt(2), (0, -1, 1): np.sqrt(2), \
                           (1, 1, 1): np.sqrt(3), (-1, -1, -1): np.sqrt(3), \
                           (1, -1, -1): np.sqrt(3), (-1, 1, -1): np.sqrt(3), (-1, -1, 1): np.sqrt(3), \
                           (1, 1, -1): np.sqrt(3), (1, -1, 1): np.sqrt(3), (-1, 1, 1): np.sqrt(3)}

    def remove_from_open(self, s: tuple) -> None:
        """
        Reconstructs the heap excluding the specified element.
        This function is needed in order to remove a specific state from the
        self.OPEN heap, as the heapq module only supports popping the smallest.
        """
        new_heap = []
        for priority, state in self.OPEN:
            if state != s:
                new_heap.append((priority, state))
        heapq.heapify(new_heap)
        self.OPEN = new_heap

    def get_g(self, s: tuple) -> float:
        """
        Get the estimated cost of the optimal path from state s to the goal.
        Set cost to inf if state is not in the dictionary.
        """
        if s not in self.g:
            self.g[s] = np.inf
        return self.g[s]

    def get_rhs(self, s: tuple) -> float:
        """
        Get the one-step lookahead cost from state s to the goal.
        Set cost to inf if state is not in the dictionary.
        """
        if s not in self.rhs:
            self.rhs[s] = np.inf
        return self.rhs[s]

    def key(self, s: tuple) -> tuple[float, float]:
        """
        Calculate and return the key for a state based on the current g and rhs values,
        adjusted by the heuristic scaled by epsilon.
        Lines 1-4 of the AD* algorithm.
        """
        if self.get_g(s) > self.get_rhs(s):
            return (self.get_rhs(s) + self.epsilon * utils.heuristic(s, self.start), self.get_rhs(s))
        else:
            return (self.get_g(s) + utils.heuristic(s, self.start), self.get_g(s))

    def update_state(self, s: tuple) -> None:
        """
        Update the g and rhs values for state s based on its neighbors.
        """
        # Line 5-6 AD* not needed due to get_g function
        if s != self.goal:  # Line 7 AD*
            self.rhs[s] = min(self.cost(s, v) + self.get_g(v) for v in self.get_neighbors(s))

        if s in [item[1] for item in self.OPEN]:  # Line 8 AD*
            self.remove_from_open(s)

        if self.get_g(s) != self.get_rhs(s):  # Line 9 AD*
            if s not in self.CLOSED:  # Line 10 AD*
                heapq.heappush(self.OPEN, (self.key(s), s))  # Line 11 AD*
                #print(f"Pushed {s} to open list with new key {self.key(s)}.")
            else:
                self.INCONS.add(s)  # Line 13 AD*

    def cost(self, u: tuple, v: tuple) -> float:
        """
        Calculate the cost of moving from node u to node v based on predefined directions.
        """
        direction = tuple(np.array(v) - np.array(u))
        return self.directions.get(direction, np.inf)  # Return inf if direction is not defined

    def get_neighbors(self, u: tuple) -> list[tuple]:
        """
        Get the neighbors of state u based on the predefined directions.
        As we are dealing with undirected graphs, the neighbors represent
        both predecessors (Pred) and successors (Succ) from the paper. 
        """
        neighbors = []
        for direction in self.directions:
            v = tuple(np.array(u) + np.array(direction))
            if utils.is_in_bound(np.array(v), self.bounds):
                neighbors.append(v)  # Add neighbors that are within the bounds
        return neighbors

    def compute_shortest_path(self) -> None:
        """
        Compute the shortest path from the start to the goal.
        """
        start_key = self.key(self.start)
        min_OPEN_key = min(s[0] for s in self.OPEN)
        while self.OPEN and (
            (min_OPEN_key[0] < start_key[0]) or 
            ((min_OPEN_key[0] == start_key[0]) and min_OPEN_key[1] < start_key[1]) or 
            self.get_rhs[self.start] != self.get_g(self.start)
        ):  # Line 14 AD*
            # Get the state with the smallest key
            _, current = heapq.heappop(self.OPEN)  # Line 15 AD*
            if current is None:
                break
            # Check for consistency
            if self.get_g(current) > self.get_rhs(current):  # Line 16 AD*
                self.g[current] = self.rhs[current]  # Line 17 AD*, make consistent
                self.CLOSED.add(current)  # Line 18 AD*
                print(current)
                for s in self.get_neighbors(current):  # Line 19 AD*
                    self.update_state(s)
            else:
                self.g[current] = np.inf  # Line 21 AD*
                for s in self.get_neighbors(current):  # Line 22 AD*
                    self.update_state(s)

    # TODO: Make MJ convex hulls accessible to path planner
    # Might be possible to do this directly through MJ functions (ray casting)
    # def _create_convex_hull(self, env):
    #     # Create a convex hull around the mesh points
    #     gateway = env.model.body('gateway_full')
    #     gateway_mesh_idx = list(range(gateway.geomadr[0], gateway.geomadr[0] + gateway.geomnum[0]))
    #     graph_adr = env.model.graphadr[gateway_mesh_idx]
    #     num_vertices = env.model.graph[graph_adr + 0]
    #     if model.
    #     hull = ConvexHull(self.mesh_points)
    #     return hull

    def generate_path(self) -> list[tuple]:
        """
        Reconstruct (sub)optimal path from start to goal based on g-values.
        """
        path = []
        s = self.start

        while s != self.goal:
            neighbors = self.get_neighbors(s)
            if not neighbors:
                break

            next_node = None
            min_cost = np.inf
            for neighbor in neighbors:
                if neighbor in self.CLOSED:
                    cost = self.cost(s, neighbor) + self.get_g(neighbor)
                    if cost < min_cost:
                        min_cost = cost
                        next_node = neighbor

            if next_node is None:
                print("Path reconstruction failed.")
                return []

            path.append(next_node)
            s = next_node

        return path


if __name__ == "__main__":
    # Create a path planner object
    planner = ADStarPlanner((0, 0, 0), (1,5,4))  # Not using env in planner yet
    start_time = time.time()
    planner.compute_shortest_path()
    print(planner.generate_path())
    print(f"Execution time: {time.time() - start_time} seconds.")
    print('done')