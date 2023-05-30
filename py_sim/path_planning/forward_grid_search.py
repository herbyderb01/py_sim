"""forward_grid_search.py: Defines a general framework for forward search
   through a grid world
"""
from abc import ABC, abstractmethod
from enum import Enum
from typing import Optional

import numpy as np
from py_sim.sensors.occupancy_grid import BinaryOccupancyGrid, ind2sub, sub2ind
from py_sim.tools.simple_priority_queue import SimplePriorityQueue


class GD(Enum):
    """Define the basic grid directions"""
    L = 1 # Left
    LU = 2 # Diagonal, left and up
    U = 3 # Up
    RU = 4 # Diagonal, right and up
    R = 5 # Right
    RD = 6 # Diagonal, right and down
    D = 7 # Down
    LD = 8 # Diagonal, left and down

class ForwardGridSearch(ABC):
    """ Defines a general framework for forward search through a grid
    """

    @abstractmethod
    def add_index_to_queue(self, ind_new: int, ind_parent: int,  direction: GD) -> None:
        """Adds the index to the queue and store parent

            Inputs:
                ind_new: Index of the new location
                ind_parent: Index of the parent node
                direction: The direction from the ind_parent cell to the ind_new cell
        """

    def resolve_duplicate(self, ind_duplicate: int, ind_poss_parent: int,  direction: GD) -> None: # pylint: disable=unused-argument
        """resolves duplicate sighting of the index - default is to do nothing

        Inputs:
            ind_duplicate: index that has been seen again
            ind_poss_parent: The possible parent that was seen in index
            duplicate
             direction is the direction of the duplicate from the parent
        """

    def __init__(self, grid: BinaryOccupancyGrid, ind_start: int, ind_end: int) -> None:
        """ Initializes the forward search by storing the grid and creating the index queue

            Inputs:
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

            Inputs:
                end_index: The index to which to plan. If None, then the
                plan end index will be used
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

    def step(self) -> tuple[int, bool]:
        """Creates one step through the while loop

            Outputs:
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
                self.visited.itemset(ind_neighbor, True)

                # Insert the node into the queue
                self.add_index_to_queue(ind_new=ind_neighbor, ind_parent=ind,  direction= direction)

        # Return the index and success flags
        return (ind, succ)


def get_neighboring_nodes(index: int, grid: BinaryOccupancyGrid) -> tuple[list[int], list[GD]]:
    """ Returns a list of neighboring nodes and the resulting directions
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
    """Defines a bread-first search through the grid"""
    def __init__(self, grid: BinaryOccupancyGrid, ind_start: int, ind_end: int) -> None:
        super().__init__(grid, ind_start, ind_end)
        self.cost: float = 1. # Variable used to implement a FIFO queue

    def add_index_to_queue(self, ind_new: int, ind_parent: int, direction: GD) -> None:
        """Adds index purely based on the number of indices in the queue to implement
           a FIFO queue

             Inputs:
                ind_new: Index of the new location
                ind_parent: Index of the parent node
                direction: The direction from the ind_parent cell to the ind_new cell
        """
        # A min queue is employed so having an always incrementing cost
        # will ensure that the next item popped off will be the item
        # that has been on the queue for the longest
        self.queue.push(cost=self.cost, index=ind_new)
        self.parent_mapping[ind_new] = ind_parent
        self.cost += 1.

class DepthFirstGridSearch(ForwardGridSearch):
    """Defines a depth-first search through the grid"""
    def __init__(self, grid: BinaryOccupancyGrid, ind_start: int, ind_end: int) -> None:
        super().__init__(grid, ind_start, ind_end)
        self.cost: float = -1. # Variable used to implement a LIFO queue

    def add_index_to_queue(self, ind_new: int, ind_parent: int, direction: GD) -> None:
        """Adds index purely based on the number of indices in the queue to implement
           a LIFO queue

             Inputs:
                ind_new: Index of the new location
                ind_parent: Index of the parent node
                direction: The direction from the ind_parent cell to the ind_new cell
        """
        # A min queue is employed so having an decrementing cost
        # will ensure that the next item popped off will be the item
        # that has been on the queue for the least amount of time
        self.queue.push(cost=self.cost, index=ind_new)
        self.parent_mapping[ind_new] = ind_parent
        self.cost -= 1.
