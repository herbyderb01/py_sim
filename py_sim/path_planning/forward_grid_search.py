"""forward_grid_search.py: Defines a general framework for forward search
   through a grid world

Attributes:
    segment_length(list[float]): Defines the length of a single grid direction movement. List is organized [L, LU, U, RU, R, RD, D, LD]
"""
from abc import ABC, abstractmethod
from enum import IntEnum
from typing import Optional, cast

import numpy as np
from py_sim.sensors.occupancy_grid import BinaryOccupancyGrid, ind2sub, sub2ind
from py_sim.tools.simple_priority_queue import SimplePriorityQueue


class GD(IntEnum):
    """Define the basic grid directions used for the search

    Attributes:
        L: Left
        LU: Diagonal, left and up
        U: Up
        RU: Diagonal, right and up
        R: Right
        RD: Diagonal, right and down
        D: Down
        LD: Diagonal, left and down
    """
    L = 0 # Left
    LU = 1 # Diagonal, left and up
    U = 2 # Up
    RU = 3 # Diagonal, right and up
    R = 4 # Right
    RD = 5 # Diagonal, right and down
    D = 6 # Down
    LD = 7 # Diagonal, left and down

# Length of single segment:    L    LU   U    RU   R    RD   D   LD
segment_length: list[float] = [1., 1.41, 1., 1.41, 1., 1.41, 1., 1.41]

class ForwardGridSearch(ABC):
    """ Defines a general framework for forward search through a grid

    Attributes:
        grid(BinaryOccupancyGrid): Occupancy grid over which to search
        queue(SimplePriorityQueue): The priority queue sorting the active nodes
        ind_start(int): The starting index value
        ind_end(int): The ending index value for the search
        parent_mapping(dict[int, int]): Stores the mapping from an index to its parent
        visited(npt.NDArray[Any]): Binary matrix indicating if a node has been visited
    """

    @abstractmethod
    def add_index_to_queue(self, ind_new: int, ind_parent: int,  direction: GD) -> None:
        """Adds the index to the queue and store parent

        Args:
            ind_new: Index of the new location
            ind_parent: Index of the parent node
            direction: The direction from the ind_parent cell to the ind_new cell
        """

    def resolve_duplicate(self, ind_duplicate: int, ind_poss_parent: int,  direction: GD) -> None: # pylint: disable=unused-argument
        """resolves duplicate sighting of the index - default is to do nothing

        Args:
            ind_duplicate: index that has been seen again
            ind_poss_parent: The possible parent that was seen in index
                duplicate
            direction: is the direction of the duplicate from the parent
        """

    def __init__(self, grid: BinaryOccupancyGrid, ind_start: int, ind_end: int) -> None:
        """ Initializes the forward search by storing the grid and creating the index queue

        Args:
            grid: Occupancy grid over which to search
            ind_start: The starting index value
            ind_end: The ending index value for the search
        """
        super().__init__()

        # Store member variables
        self.grid = grid
        self.queue = SimplePriorityQueue()
        self.queue.push(cost=0., index=ind_start)
        self.ind_start = ind_start
        self.ind_end = ind_end

        # Create indexing variables
        self.parent_mapping: dict[int, int] = {} # Stores the mapping from an index to its parent
        self.visited = np.zeros((self.grid.n_rows, self.grid.n_cols), dtype=bool)

        # Evaluate the start and end index to ensure they are on the map
        if ind_start > grid.max_index or ind_end > grid.max_index:
            raise ValueError("Indices must be located on the grid")

    def get_plan(self, end_index: Optional[int] = None) -> list[int]:
        """Returns a list of indices from the start index to end_index. Assumes that
           planning has already been performed. Throws a ValueError if the end index cannot be connected to the starting index

            Args:
                end_index: The index to which to plan. If None, then the
                plan end index will be used

            Returns:
                list[int]: Indices of the plan from start to end
        """
        # Create the index from which to start the search
        if end_index is None:
            ind = self.ind_end
        else:
            ind = end_index

        # Search the tree for the result
        plan: list[int] = [ind]
        while ind != self.ind_start:
            if ind in self.parent_mapping:
                ind = self.parent_mapping[ind]
                plan.append(ind)
            else:
                raise ValueError("No path from end index to start")

        # Reverse the plan list so that it goes from start to end
        plan.reverse()
        return plan

    def get_plan_cartesian(self, end_index: Optional[int] = None) -> tuple[list[float], list[float]]:
        """Returns the (x,y) cartesian coordinates of each point along the plan. Assumes that
           planning has already been performed. Throws a ValueError if the end index cannot be connected to the starting index

            Args:
                end_index: The index to which to plan. If None, then the plan end index will be used

            Returns:
                tuple[list[float], list[float]]: A list of x and y coordinates for each point along the plan
        """
        # Get the plan indices
        indices = self.get_plan(end_index=end_index)

        # Loop through and convert each of the plan indices into cartesian coordinates
        x_vec: list[float] = []
        y_vec: list[float] = []
        for ind in indices:
            # Get the corresponding row and column
            (row, col) = ind2sub(n_cols=self.grid.n_cols, ind=ind)

            # Get the position
            position = self.grid.indices_to_position(row=row, col=col)
            x_vec.append(position.x)
            y_vec.append(position.y)

        # Return the resulting plan
        return (x_vec, y_vec)

    def calculate_plan_length(self, plan: Optional[list[int]]= None) -> float:
        """ Calculates the length of a plan. If the plan is not provided then
            it is calculated using get_plan with self.ind_end

        Args:
            plan: Optional plan for which the length will be calculated

        Returns:
            float: Plan length
        """
        # Get the plan
        if plan is None:
            plan = self.get_plan() # Stores the plan indices

        # Initialize the calculation variables
        plan_length = 0. # Iteratively calculated plan length
        ind_prev = self.ind_start # Stores the index to the previous point
        q_prev = self.grid.index_to_position(ind=ind_prev)

        # Loop through the plan and calculate the point to point length
        for ind in plan:
            # Get the position corresponding to the index
            q_new = self.grid.index_to_position(ind=ind)

            # Update the length
            plan_length += float(np.linalg.norm(q_new.state - q_prev.state))

            # Update for the next iteration
            q_prev = q_new

        return plan_length

    def search(self) -> bool:
        """Searches the grid from start to end and returns true if successful"""
        # Search until the goal has been found
        goal_found = False
        while not goal_found:
            # Perform a step
            (ind, goal_found) = self.step()

            # Break if the index negative
            if ind < 0:
                break

        return goal_found

    def complete_search(self) -> None:
        """Searches the grid from the start to end, exhaustively searching until the entire grid has been covered
        """
        ind = 0
        while ind >= 0: # Loops until ind = -1, which occurs when the queue is empty
            # Perform a step
            ind, _ = self.step()

    def step(self) -> tuple[int, bool]:
        """Creates one step through the while loop

        Returns:
            tuple[int, bool]:
                ind: The index of the node that was visited
                        -1 => that there are no elements in the queue
                succ: true => goal reached, false => it was not
        """

        # Initialize the outputs
        ind = -1
        succ = False
        if self.queue.count() <= 0:
            return (ind, succ)
        ind = self.queue.pop()

        # Check to see if the final goal has been reached
        succ = bool(ind == self.ind_end)

        # Get the neighbors of ind
        [ind_neighbors, dir_neighbors] = get_neighboring_nodes(index=ind, grid=self.grid)

        # Process each of the neighbors
        for (ind_neighbor,  direction) in zip(ind_neighbors, dir_neighbors):
            # Resolve duplicates
            if self.visited.item(ind_neighbor):
                self.resolve_duplicate(ind_duplicate=ind_neighbor, ind_poss_parent=ind,  direction= direction)

            # Process a node that has not yet been visited
            else:
                # Mark as visited
                np.put(self.visited, ind_neighbor, True)

                # Insert the node into the queue
                self.add_index_to_queue(ind_new=ind_neighbor, ind_parent=ind,  direction= direction)

                # Update the parent relationship
                self.parent_mapping[ind_neighbor] = ind

        # Return the index and success flags
        return (ind, succ)

def get_neighboring_nodes(index: int, grid: BinaryOccupancyGrid) -> tuple[list[int], list[GD]]:
    """ Returns a list of neighboring nodes and the resulting directions

    Args:
        index: grid index for which the neighbors are returned
        grid: Occupancy grid defining obstacle locations in the grid

    Returns:
        tuple[list[int], list[GD]]: list of node indices and corresponding list of directions
    """
    # Initialize outputs
    indices: list[int] = []
    dirs: list[GD] = []

    # Check whether the index corresponds to far left or right
    row_ind, col_ind = ind2sub(n_cols=grid.n_cols, ind=index)

    # Offsets
    row_offsets = [0,    -1,    1,    -1,    1,    0,    -1,    1]
    col_offsets = [-1,   -1,    -1,    0,    0,    1,     1,    1]
    directions =  [GD.L, GD.LU, GD.LD, GD.U, GD.D, GD.R, GD.RU, GD.RD]

    # Calculate all valid directions
    for (row_off, col_off,  direction) in zip(row_offsets, col_offsets, directions):
        # Get the indices
        row = row_ind + row_off
        col = col_ind + col_off

        # Evaluate whether the location is free of obstacles
        if grid.free_subscripts(row=row, col=col):
            indices.append(sub2ind(n_cols=grid.n_cols, row_ind=row, col_ind=col))
            dirs.append( direction)

    # Return the results
    return (indices, dirs)

class BreadFirstGridSearch(ForwardGridSearch):
    """Defines a bread-first search through the grid

    Attributes:
        cost(float): Variable used to implement a FIFO queue. Stores the cost of the next element.
    """
    def __init__(self, grid: BinaryOccupancyGrid, ind_start: int, ind_end: int) -> None:
        super().__init__(grid, ind_start, ind_end)
        self.cost: float = 1. # Variable used to implement a FIFO queue

    def add_index_to_queue(self, ind_new: int, ind_parent: int, direction: GD) -> None:
        """Adds index purely based on the number of indices in the queue to implement
           a FIFO queue

        Args:
            ind_new: Index of the new location
            ind_parent: Index of the parent node
            direction: The direction from the ind_parent cell to the ind_new cell
        """
        # A min queue is employed so having an always incrementing cost
        # will ensure that the next item popped off will be the item
        # that has been on the queue for the longest
        self.queue.push(cost=self.cost, index=ind_new)
        self.cost += 1. # Update the cost for the next element to be added for a LIFO queue

class DepthFirstGridSearch(ForwardGridSearch):
    """Defines a depth-first search through the grid

    Attributes:
        cost(float): Variable used to implement a FIFO queue. Stores the cost of the next element.
    """
    def __init__(self, grid: BinaryOccupancyGrid, ind_start: int, ind_end: int) -> None:
        super().__init__(grid, ind_start, ind_end)
        self.cost: float = -1. # Variable used to implement a LIFO queue

    def add_index_to_queue(self, ind_new: int, ind_parent: int, direction: GD) -> None:
        """Adds index purely based on the number of indices in the queue to implement
           a LIFO queue

        Args:
            ind_new: Index of the new location
            ind_parent: Index of the parent node
            direction: The direction from the ind_parent cell to the ind_new cell
        """
        # A min queue is employed so having an decrementing cost
        # will ensure that the next item popped off will be the item
        # that has been on the queue for the least amount of time
        self.queue.push(cost=self.cost, index=ind_new)
        self.cost -= 1. # Update the cost for the next element to be added for a LIFO queue

class DijkstraGridSearch(ForwardGridSearch):
    """Defines a cost-wavefront search through the grid

    Attributes:
        c2c(npt.NDArray[Any]): Grid storing the value of the cost to come to each node
    """
    def __init__(self, grid: BinaryOccupancyGrid, ind_start: int, ind_end: int) -> None:
        super().__init__(grid, ind_start, ind_end)

        # Create a storage container for cost-to-come
        self.c2c = np.full(grid.grid.shape, np.inf, dtype=float)
        np.put(self.c2c, ind_start, 0.)

    def add_index_to_queue(self, ind_new: int, ind_parent: int, direction: GD) -> None:
        """Adds index to queue based on the cost to come to that new index.

        Args:
            ind_new: Index of the new location
            ind_parent: Index of the parent node
            direction: The direction from the ind_parent cell to the ind_new cell
        """
        # Calculate the cost to come as the cost of the parent plus the edge cost
        cost = segment_length[direction] + self.c2c.item(ind_parent)

        # Add the node to the list
        np.put(self.c2c, ind_new, cost)
        self.queue.push(cost=cost, index=ind_new)

    def resolve_duplicate(self, ind_duplicate: int, ind_poss_parent: int,  direction: GD) -> None:
        """resolves duplicate sighting of the index - checks to see if the lowest cost to come
           to the node has a lower cost-to-come than the previous path found, if so then the
           cost is updated

        Args:
            ind_duplicate: index that has been seen again
            ind_poss_parent: The possible parent that was seen in index
                duplicate
            direction: is the direction of the duplicate from the parent
        """
        # Calculate the cost-to-come of the new possible edge
        cost_possible = segment_length[direction] + self.c2c.item(ind_poss_parent)

        # Update the parentage and cost-to-come if a lower cost route has been found
        # Note that a small number is subtracted from the previous cost to avoid
        # updates due to small numerical precision
        if cost_possible < self.c2c.item(ind_duplicate)-1e-5:
            self.queue.update(cost=cost_possible, index=ind_duplicate)
            np.put(self.c2c, ind_duplicate, cost_possible)
            self.parent_mapping[ind_duplicate] = ind_poss_parent

class AstarGridSearch(ForwardGridSearch):
    """Defines a cost-wavefront search through the grid

    Attributes:
        c2c(npt.NDArray[Any]): Grid storing the value of the cost to come to each node
    """
    def __init__(self, grid: BinaryOccupancyGrid, ind_start: int, ind_end: int) -> None:
        super().__init__(grid, ind_start, ind_end)

        # Create a storage container for cost-to-come
        self.c2c = np.full(grid.grid.shape, np.inf, dtype=float)
        np.put(self.c2c, ind_start, 0.)

        # Convert the end index to matrix indexing
        row_goal, col_goal = ind2sub(n_cols=grid.n_cols, ind=ind_end)
        self.end_ind_mat = np.array([[row_goal],[col_goal]])

    def cost_to_go_heuristic(self, ind: int) -> float:
        """Calculates a heuristic on the cost to go from the index in question

        Args:
            ind: grid index in question

        Returns:
            float: Euclidean grid distance from ind to the goal index
        """
        # Get an array of the row/column indices
        row, col = ind2sub(n_cols=self.grid.n_cols, ind=ind)
        ind_mat = np.array([[row], [col]])

        # Return the distance
        return  cast(float, np.linalg.norm(ind_mat - self.end_ind_mat))

    def add_index_to_queue(self, ind_new: int, ind_parent: int, direction: GD) -> None:
        """Adds index to queue based on the cost-to-come to that new index plus the hueristic cost-to-go from that index to the goal

        Args:
            ind_new: Index of the new location
            ind_parent: Index of the parent node
            direction: The direction from the ind_parent cell to the ind_new cell
        """
        # Calculate the cost to come as the cost of the parent plus the edge cost
        print("Fix me!!!")
        c2c = segment_length[direction] + self.c2c.item(ind_parent)
        cost_heuristic = 0.

        # Add the node to the list
        print("Fix me!!!")
        np.put(self.c2c, ind_new, c2c)
        self.queue.push(cost=0., index=ind_new)

    def resolve_duplicate(self, ind_duplicate: int, ind_poss_parent: int,  direction: GD) -> None:
        """resolves duplicate sighting of the index - checks to see if the lowest cost-to-come plus heuristic
           a lower value than the previous path found, if so then the
           cost is updated

        Args:
            ind_duplicate: index that has been seen again
            ind_poss_parent: The possible parent that was seen in index
                duplicate
            direction: the direction of the duplicate from the parent
        """
        # Calculate the cost-to-come of the new possible edge
        c2c_possible = segment_length[direction] + self.c2c.item(ind_poss_parent)

        # Update the parentage and cost-to-come if a lower cost route has been found
        # Note that a small number is subtracted from the previous cost to avoid
        # updates due to small numerical precision
        if c2c_possible < self.c2c.item(ind_duplicate)-1e-5:
            print("Fix me!!!")
            cost_heuristic = 0.

            self.queue.update(cost=cost_heuristic, index=ind_duplicate)
            np.put(self.c2c, ind_duplicate, c2c_possible)
            self.parent_mapping[ind_duplicate] = ind_poss_parent

class GreedyGridSearch(ForwardGridSearch):
    """Defines a cost-wavefront search through the grid

    Attributes:
        c2c(npt.NDArray[Any]): Grid storing the value of the cost to come to each node
    """
    def __init__(self, grid: BinaryOccupancyGrid, ind_start: int, ind_end: int) -> None:
        super().__init__(grid, ind_start, ind_end)

        # Create a storage container for cost-to-come
        self.c2c = np.full(grid.grid.shape, np.inf, dtype=float)
        np.put(self.c2c, ind_start, 0.)

        # Convert the end index to matrix indexing
        row_goal, col_goal = ind2sub(n_cols=grid.n_cols, ind=ind_end)
        self.end_ind_mat = np.array([[row_goal],[col_goal]])

    def cost_to_go_heuristic(self, ind: int) -> float:
        """Calculates a heuristic on the cost to go from the index in question

        Args:
            ind: grid index in question

        Returns:
            float: Euclidean grid distance from ind to the goal index
        """
        # Get an array of the row/column indices
        row, col = ind2sub(n_cols=self.grid.n_cols, ind=ind)
        ind_mat = np.array([[row], [col]])

        # Return the distance
        return  0.

    def add_index_to_queue(self, ind_new: int, ind_parent: int, direction: GD) -> None:
        """Adds index to queue based on the cost to come to that new index.

        Args:
            ind_new: Index of the new location
            ind_parent: Index of the parent node
            direction: The direction from the ind_parent cell to the ind_new cell
        """
        # Calculate the cost to come as the cost of the parent plus the edge cost
        c2c = segment_length[direction] + self.c2c.item(ind_parent)
        np.put(self.c2c, ind_new, c2c)

        # Add the node to the list using the cost-to-go heuristic
        print("Fix me!!!")
        cost_heuristic = 0.
        self.queue.push(cost=0., index=ind_new)

    def resolve_duplicate(self, ind_duplicate: int, ind_poss_parent: int,  direction: GD) -> None:
        """resolves duplicate sighting of the index - checks to see if the lowest cost to come
           to the node has a lower cost-to-come than the previous path found, if so then the
           cost is updated

        Args:
            ind_duplicate: index that has been seen again
            ind_poss_parent: The possible parent that was seen for index
                duplicate
            direction: the direction of the duplicate from the parent
        """
        # Calculate the cost-to-come of the new possible edge
        print("Fix me!!!")
        c2c_possible = 0.

        # Update the parentage and cost-to-come if a lower cost route has been found
        # Note that a small number is subtracted from the previous cost to avoid
        # updates due to small numerical precision
        if c2c_possible < self.c2c.item(ind_duplicate)-1e-5:
            print("Fix me!!!")
