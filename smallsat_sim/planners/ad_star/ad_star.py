import numpy as np
from smallsat_sim.planners.ad_star import utils
from smallsat_sim.planners.base_planner import BasePlanner
import heapq
import time
import threading
import mujoco


class ADStarPlanner(BasePlanner):
    """
    This class implements the Anytime Dynamic A* (AD*) path planning algorithm.
    It is based on:
    "Maxim Likhachev, David Ferguson, Geoff Gordon, Anthony Stentz, and Sebastian Thrun. (2005)
    “Anytime Dynamic A*: An Anytime, Replanning Algorithm.” In Proc. Int. Conf. Automated Planning and Scheduling, 15. 
    https://aaai.org/papers/icaps-05-027-anytime-dynamic-a-an-anytime-replanning-algorithm/."

    @author: Markus H. Iversflaten
    """
    def __init__(self, env) -> None:  # Not using env in planner yet
        super().__init__(env)
        self.env = env
        # Resolution of discrete grid
        self.resolution = env.env_cfg.planner.resolution
        # Scaling factor for the heuristic (inflation factor)
        self.epsilon = env.env_cfg.planner.epsilon
        self.epsilon_increment = env.env_cfg.planner.epsilon_increment
        self.epsilon_decrement = env.env_cfg.planner.epsilon_decrement
        self.bounds = env.env_cfg.planner.bounds  # Bounds of the environment
        # Define the possible directions and their costs (with scale)
        self.directions = self._scale_directions(env.env_cfg.planner.unit_directions)
        self.start = env.env_cfg.planner.start_pos  # Initial position of the agent
        self.goal = env.env_cfg.planner.goal_pos  # Goal position of the agent

        # Inconsistent states "s": Overconsistent: g(s) > rhs(s), Underconsistent: g(s) < rhs(s)
        self.OPEN = []  # Priority queue of inconsistent states to be expanded
        self.OPEN_SET = set()  # Set of states that are in the priority queue
        self.g = {self.start: np.inf, self.goal: np.inf}  # Dictionary of costs from each state to goal
        self.rhs = {self.start: np.inf, self.goal: 0}  # Dictionary of one-step lookahead costs

        # Insert the goal into the priority queue to start the expansion
        heapq.heappush(self.OPEN, (self._key(self.goal), self.goal))
        self.OPEN_SET.add(self.goal)
        self.CLOSED = set()  # Set of states that have been expanded
        self.INCONS = set()  # Set of states that have been expanded and are inconsistent

        # Define the colors for the path visualization
        self.colors = np.array([
            [0.0, 1.0, 0.0, 1.0],  # Green
            [1.0, 0.0, 0.0, 1.0],  # Red
            [0.0, 0.0, 1.0, 1.0],  # Blue
            [1.0, 1.0, 0.0, 1.0],  # Yellow
            [0.0, 1.0, 1.0, 1.0],  # Cyan
            [1.0, 0.0, 1.0, 1.0],  # Magenta
            [0.5, 0.5, 0.5, 1.0],  # Gray
            [1.0, 0.5, 0.0, 1.0],  # Orange
            [0.5, 0.0, 0.5, 1.0],  # Purple
            [0.0, 0.5, 0.5, 1.0]   # Teal
        ])
        self.num_colors = 10
        self.color_idx = 0  # Visualize different paths with different colors

        self.path = [(0,0,10)]  # List of states that form the optimal path
        self.idx_reference_point = 0  # Index of the current reference point
        
        self.start_planning_thread()
    
    def _scale_directions(self, unit_directions) -> dict:
        """
        Scale the directions based on the resolution of the grid.
        """
        scaled_directions = {
            tuple(np.array(key) * self.resolution): value * self.resolution
            for key, value in unit_directions.items()
        }
        return scaled_directions

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
            else:
                self.INCONS.add(s)  # Line 13 AD*

    def _cost(self, u: tuple, v: tuple) -> float:
        """
        Calculate the cost of moving from node u to node v based on predefined directions.
        """
        return utils.get_distance(u, v)

    def _get_neighbors(self, u: tuple) -> list[tuple]:
        """
        Get the neighbors of state u based on the predefined directions.
        As we are dealing with undirected graphs, the neighbors represent
        both predecessors (Pred) and successors (Succ) from the paper. 
        """
        neighbors = []
        for direction in self.directions:
            v = tuple(np.array(u) + np.array(direction))
            if utils.is_in_bound(np.array(v), self.bounds):
                if not self._check_collision(u, direction):  # Check for collision
                    # Add valid neighbor
                    neighbors.append(v)
        return neighbors
    
    def _check_collision(self, point: np.ndarray, direction) -> tuple[bool, float]:
        """
        Checks if a ray from a point in a certain direction 
        intersects with an obstacle.

        Returns distance to collision or -1 if no intersection.
        """
        _ = np.zeros((1, 1), dtype=np.int32)  # Pointer that is needed but unused
        body_exclude_id = 2  # Exclude the SmallSat body from collision check
        flg_static = 1  # 0: exclude static geoms, 1: include static geoms
        group_exclusion = None  # Exclude no geom groups
        point = np.array(point, dtype=np.float64)  # Cast to float64
        direction = np.array(direction, dtype=np.float64)  # Cast to float64
        # Calculate distance to collision
        dist = mujoco.mj_ray(self.env.model, self.env.data, point, direction,
                             group_exclusion, flg_static, body_exclude_id, _)
        
        # Check if collision is imminent
        if dist < 0 or dist > self.resolution:
            return False
        else:
            return True
    
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
                for s in self._get_neighbors(current):  # Line 19 AD*
                    self._update_state(s)
            else:
                self.g[current] = np.inf  # Line 21 AD*
                self._update_state(current)  # Part of line 22 (union of neighbors and current)
                for s in self._get_neighbors(current):  # Line 22 AD*
                    self._update_state(s)

    def generate_path(self) -> list[tuple]:
        """
        Reconstruct (sub)optimal path from start to goal based on g-values.
        """
        path = []
        s = self.start
        visited = set()
        i = 0

        while utils.get_distance(s, self.goal) > self.resolution:
            neighbors = self._get_neighbors(s)
            if not neighbors:
                print("No neighbors found.")
                break

            next_node = None
            min_cost = np.inf
            for neighbor in neighbors:
                if neighbor not in visited:
                    cost = self._cost(s, neighbor) + self._get_g(neighbor)
                    if cost < min_cost:
                        min_cost = cost
                        next_node = neighbor

            if next_node is None:
                print("Path reconstruction failed.")
                return []

            path.append(next_node)
            s = next_node
            if i > 100:
                print("Path reconstruction failed.")
                return []
            i += 1

        return path
    
    def plan(self):
        """
        Main planning loop for the AD* algorithm.
        """
        while self.path == []:  # Initial planning
            self.compute_shortest_path()
            self.path = self.generate_path()
            if self.path == []:
                self.epsilon -= self.epsilon_decrement
                continue
        prev_path = self.path

        while True:  # TODO: Keep planner loop ready for replanning
            if False:  # TODO: if replanning is needed
                self.epsilon += self.epsilon_increment
            elif self.epsilon > 1:
                # Limit epsilon to 1 from below
                self.epsilon = max(1, self.epsilon - self.epsilon_decrement)
            # Move states from INCONS to OPEN
            for s in list(self.INCONS):
                self.INCONS.remove(s)
                heapq.heappush(self.OPEN, (self._key(s), s))
                self.OPEN_SET.add(s)

            # Update keys in the OPEN list with the new epsilon value
            self._update_open_keys()

            self.CLOSED = set()
            self.compute_shortest_path()
            self.color_idx += 1
            self.path = self.generate_path()
            if self.path == []:  # If failure, keep previous path
                self.path = prev_path
            if self.epsilon <= 1:  # Optimal path, break afterwards
                self.color_idx += 1
                self.path = self.generate_path()
                break

    def start_planning_thread(self):
        self.planning_thread = threading.Thread(target=self.plan)
        self.planning_thread.start()

    def stop_planning_thread(self):
        if self.planning_thread.is_alive():
            self.planning_thread.join()

    def get_reference(self, obs: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(
            obs[0:3]
            - self.path[
                self.idx_reference_point % len(self.path)
            ]
        )

        color = self.colors[self.color_idx % self.num_colors]
        self.visualize([np.array(t) for t in self.path[self.idx_reference_point:]], color=color)

        # If smallsat is closer than the clearance distance, the next reference point is queried
        if dist < 0.2:
            self.idx_reference_point += 1

        return np.array(self.path[
            self.idx_reference_point % len(self.path)
        ])


if __name__ == "__main__":
    from smallsat_sim.envs.astrobee.env import AstrobeeEnv
    from smallsat_sim.utils.helpers import get_args
    import matplotlib.patches as patches
    import matplotlib.pyplot as plt
    args = get_args()
    # Create a path planner object
    #env = AstrobeeEnv(args)


    planner = ADStarPlanner((0, 0, 0), (9, 9, 9))  # Create planner object
    
    print('start')
    planner.compute_shortest_path()
    path = planner.generate_path()
    # Main planning loop
    while True:
        previous_path = path
        if False:  # TODO: if replanning is needed 
            planner.epsilon += 0.2
        elif planner.epsilon > 1:
            planner.epsilon = max(1, planner.epsilon - 1)  # Upper limit epsilon to 1
        # Move states from INCONS to OPEN
        for s in list(planner.INCONS):
            planner.INCONS.remove(s)
            heapq.heappush(planner.OPEN, (planner._key(s), s))
            planner.OPEN_SET.add(s)
        
        # Update keys in the OPEN list with the new epsilon value
        planner._update_open_keys()

        planner.CLOSED = set()
        planner.compute_shortest_path()
        start = time.time()
        path = planner.generate_path()
        print(f"Execution time: {time.time() - start} seconds.")
        if path != previous_path:
            print('changed path')
        print(path)
        #print(planner.epsilon)
        if planner.epsilon <= 1:  # Optimal path has been found
            #path = planner.generate_path()
            print(len(path))
            print(path)
            break
