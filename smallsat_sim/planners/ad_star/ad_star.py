import numpy as np
from smallsat_sim.planners.ad_star import utils
import heapq
import time
import threading


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
        self.OPEN_SET = set()  # Set of states that are in the priority queue
        self.g = {start: np.inf, goal: np.inf}  # Dictionary of costs from each state to goal
        self.rhs = {start: np.inf, goal: 0}  # Dictionary of one-step lookahead costs
        self.epsilon = 3.0  # Scaling factor for the heuristic (inflation factor)
        self.resolution = 1  # Resolution of the grid
        self.bounds = np.array([[0, 0, 0], [10, 10, 10]])  # Hardcoded bounds for the environment

        # Insert the goal into the priority queue to start the expansion
        heapq.heappush(self.OPEN, (self._key(self.goal), self.goal))
        self.OPEN_SET.add(self.goal)
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

        # Define preliminary obstacles in the environment
        self.obstacles = self.define_obstacles()
        
    def define_obstacles(self) -> set:
        """
        Define obstacles that cover multiple nodes and return a set of obstacle coordinates.
        """
        obstacles = set()

        # Example 1: Cubic obstacle from (2,2,2) to (4,4,4)
        for x in range(2, 5):
            for y in range(2, 5):
                for z in range(2, 5):
                    obstacles.add((x, y, z))

        # Example 2: Rectangular obstacle from (5,0,0) to (5,9,9)
        for x in range(5, 6):
            for y in range(0, 10):
                for z in range(0, 10):
                    obstacles.add((x, y, z))

        # Example 3: Line obstacle from (8,0,0) to (8,9,0)
        for y in range(0, 10):
            obstacles.add((8, y, 0))

        # Add more obstacle definitions as needed

        return obstacles

    def _remove_from_open(self, s: tuple) -> None:
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
        self.OPEN_SET.remove(s)

    def _get_g(self, s: tuple, default: float = np.inf) -> float:
        """
        Get the estimated cost of the optimal path from state s to the goal.
        Return a deafult value if the key does not exist.
        """
        return self.g.get(s, default)

    def _get_rhs(self, s: tuple, default: float = np.inf) -> float:
        """
        Get the one-step lookahead cost from state s to the goal.
        Return a deafult value if the key does not exist.
        """
        return self.rhs.get(s, np.inf)

    def _key(self, s: tuple) -> tuple[float, float]:
        """
        Calculate and return the key for a state based on the current g and rhs values,
        adjusted by the heuristic scaled by epsilon.
        Lines 1-4 of the AD* algorithm.
        """
        if self._get_g(s) > self._get_rhs(s):
            return (self._get_rhs(s) + self.epsilon * utils.heuristic(s, self.start), self._get_rhs(s))
        else:
            return (self._get_g(s) + utils.heuristic(s, self.start), self._get_g(s))

    def _update_state(self, s: tuple) -> None:
        """
        Update the g and rhs values for state s based on its neighbors.
        """
        if s not in self.CLOSED:  # Line 5 AD*
            self.g[s] = np.inf  # Line 6 AD*
        if s != self.goal:  # Line 7 AD*
            self.rhs[s] = min([self._cost(s, v) + self._get_g(v) for v in self._get_neighbors(s)])

        if s in self.OPEN_SET:  # Line 8 AD*
            self._remove_from_open(s)

        if self._get_g(s) != self._get_rhs(s):  # Line 9 AD*
            if s not in self.CLOSED:  # Line 10 AD*
                heapq.heappush(self.OPEN, (self._key(s), s))  # Line 11 AD*
                self.OPEN_SET.add(s)
                #print(f"Pushed {s} to open list with new key {self.key(s)}.")
            else:
                self.INCONS.add(s)  # Line 13 AD*

    def _cost(self, u: tuple, v: tuple) -> float:
        """
        Calculate the cost of moving from node u to node v based on predefined directions.
        """
        direction = tuple(np.array(v) - np.array(u))
        return self.directions.get(direction, np.inf)  # Return inf if direction is not defined

    def _get_neighbors(self, u: tuple) -> list[tuple]:
        """
        Get the neighbors of state u based on the predefined directions.
        As we are dealing with undirected graphs, the neighbors represent
        both predecessors (Pred) and successors (Succ) from the paper. 
        """
        neighbors = []
        for direction in self.directions:
            v = tuple(np.array(u) + np.array(direction))
            if utils.is_in_bound(np.array(v), self.bounds) and v not in self.obstacles:
                neighbors.append(v)  # Add neighbors that are within the bounds and not in obstacles
        return neighbors
    
    def _update_open_keys(self):
        """
        Update the keys for all states in the OPEN list with the new epsilon value.
        """
        new_open = []
        while self.OPEN:
            _, state = heapq.heappop(self.OPEN)
            new_open.append((self._key(state), state))
        heapq.heapify(new_open)
        self.OPEN = new_open
        self.OPEN_SET = {state for _, state in self.OPEN}  # Update the set as well

    def compute_shortest_path(self) -> None:
        """
        Compute the shortest path from the start to the goal.
        """
        while self.OPEN and (self.OPEN[0][0] < self._key(self.start) or 
            self._get_rhs(self.start) != self._get_g(self.start)
        ):  # Line 14 AD*
            # Get the state with the smallest key
            _, current = heapq.heappop(self.OPEN)  # Line 15 AD*
            if current is None:
                break
            # Check for consistency
            if self._get_g(current) > self._get_rhs(current):  # Line 16 AD*
                self.g[current] = self.rhs[current]  # Line 17 AD*, make consistent
                self.CLOSED.add(current)  # Line 18 AD*
                #print(current)
                for s in self._get_neighbors(current):  # Line 19 AD*
                    self._update_state(s)
            else:
                self.g[current] = np.inf  # Line 21 AD*
                for s in self._get_neighbors(current):  # Line 22 AD*
                    self._update_state(s)

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
        visited = set()

        while s != self.goal:
            neighbors = self._get_neighbors(s)
            if not neighbors:
                print("No neighbors found.")
                break

            next_node = None
            min_cost = np.inf
            for neighbor in neighbors:
                cost = self._cost(s, neighbor) + self._get_g(neighbor)
                if cost < min_cost and neighbor not in visited:
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
    planner = ADStarPlanner((0, 0, 0), (9, 9, 9))  # Not using env in planner yet
    #start_time = time.time()
    planner.compute_shortest_path()
    print(len(planner.generate_path()))
    while True:
        if False:  # TODO: if replanning is needed 
            planner.epsilon += 0.2
        elif planner.epsilon > 1:
            planner.epsilon = max(1, planner.epsilon - 0.2)
        # Move states from INCONS to OPEN
        for s in list(planner.INCONS):
            planner.INCONS.remove(s)
            heapq.heappush(planner.OPEN, (planner._key(s), s))
            planner.OPEN_SET.add(s)
        
        # Update keys in the OPEN list with the new epsilon value
        planner._update_open_keys()

        planner.CLOSED = set()
        planner.compute_shortest_path()
        print(planner.epsilon)
        if planner.epsilon <= 1.1:
            print(len(planner.generate_path()))
            break

    # print(planner.generate_path())
    # print(f"Execution time: {time.time() - start_time} seconds.")
    # print('done')