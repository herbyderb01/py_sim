"""occupancy_grid.py Stores the classes and defs for interacting with an occupancy grid
"""

from typing import cast

import numpy as np
from py_sim.tools.sim_types import TwoDimArray
from py_sim.worlds.polygon_world import PolygonWorld


class BinaryOccupancyGrid:
    """Stores a binary occupancy grid
        Constants:
            FREE:  Boolean indicating that a location is free of obstacles
            OCCUPIED:  Boolean indicating that a location is occupied

        Properties:
            res: Resolution of the occupancy grid
            x_lim: min and max x-values
            y_lim: min and max y-values
            tl_corner: Location of the top-left corner
            n_cols: Number of columns in the grid
            n_rows: Number of rows in the grid
            grid: Binary grid storing the occupancy

    """
    FREE: bool = False # Boolean indicating that a location is free of obstacles
    OCCUPIED: bool = True # Boolean indicating that a location is occupied
    def __init__(self, res: float, x_lim: tuple[float, float], y_lim: tuple[float, float]) -> None:
        """Initializes the occupance grid values

            Inputs:
                res: Resolution of the occupancy grid
                x_lim: min and max x-values
                y_lim: min and max y-values
        """
        # Initialize properties
        self.res = np.abs(res)                  # Resolution of the grid
        self.res_half = self.res/2              # Half of the resolution
        self.tl_corner = TwoDimArray(x=x_lim[0], # Location of the top-left corner
                                     y=y_lim[1]) # Corresponds to minimum x-value and maximum y

        # Create the grid
        x_diff = x_lim[1]-x_lim[0]
        self.n_cols: int = int(np.ceil(x_diff/res))  # Number of columns in the grid
        y_diff = y_lim[1]-y_lim[0]
        self.n_rows: int = int(np.ceil(y_diff/res))  # Number of rows in the grid
        self.grid = np.zeros((self.n_rows, self.n_cols), dtype=bool) # Binary grid
        self.x_lim = x_lim                      # x and y limits of the grid
        self.y_lim = y_lim

    def isOccupied(self, q: TwoDimArray) -> bool:
        """ isOccupied returns true if the position is in an occupied
            cell, false otherwise. An invalid position is considered occupied

            Inputs:
                q: 2x1 position

            Returns:
                False if free, true otherwise
        """

        # Calculate the indices
        (row, col, valid) = self.positionToIndex(q)

        # Determine if it is occupied
        if valid:
            return cast(bool, self.grid[row, col] == self.OCCUPIED)
        return True # Default to occupied


    def setOccupied(self, q: TwoDimArray) -> None:
        """ setOccupied sets the grid location corresponding to the 2x1
            position q as occupied. Does nothing if q is out of bounds

            Inputs:
                q: 2x1 position
        """

        # Calculate the indices
        (row, col, valid) = self.positionToIndex(q)

        # Set cell as occupied
        if valid:
            self.grid[row, col] = self.OCCUPIED

    def setFree(self, q: TwoDimArray) -> None:
        """ setFree sets the grid location corresponding to the 2x1
            position q as free

            Inputs:
                q: 2x1 position
        """
        # Calculate the indices
        (row, col, valid) = self.positionToIndex(q)

        # Set cell as Free
        if valid:
            self.grid[row, col] = self.FREE

    def positionToIndex(self, q: TwoDimArray) -> tuple[int, int, bool]:
        """ positionToIndex converts a position into an index

            Inputs:
                q: 2x1 position corresponding to the point

            Outputs:
                row: row index inside grid
                col: column index inside grid
                Valid: boolean indicating whether the passed in position is valid (True)
        """
        # Translate the position
        q_trans = q.state - self.tl_corner.state

        # Divide by resolution to get index
        q_ind = q_trans / self.res

        # Round to get integer
        col_unsat = np.round(q_ind.item(0) + self.res_half)
        row_unsat = -np.round(q_ind.item(1) - self.res_half)   # Negative sign comes from top-left
                                                               # being the max y value (positive in
                                                               # row moves negative in y)
        # Saturate to ensure on map
        row = max(0, row_unsat)
        row = min(self.n_rows-1, row)
        col = max(0, col_unsat)
        col = min(self.n_cols-1, col)

        # Determine if the position is valid in the map
        valid = bool(row == row_unsat and col == col_unsat)

        return (row, col, valid)

    def indexToPosition(self, row: int, col: int) -> TwoDimArray:
        """ indexToPosition converts an index into a position

            Inputs:
                row: row index inside grid
                col: column index inside grid

            Outputs:
                q: 2x1 position corresponding to the point
        """
        # Create the position
        # Negative sign comes from tl being the max y value (positive in row moves negative in y)
        q = self.tl_corner.state + np.array([[self.res*col-self.res_half],
                                             [-self.res*row+self.res_half]])

        return TwoDimArray(x=q.item(0), y=q.item(1))

    def get_cell_box(self, row: int, col: int) -> tuple[list[float, float, float, float],
                                                        list[float, float, float, float]]:
        """ Defines the bounding box for a cell given by the row and column indices

            Inputs:
                row: Index for the grid row
                col: Index for the grid column

            Outputs:
                (x_vec, y_vec): Define the coordinates of the cell corners
        """
        # Initailize the outputs
        x = [0., 0., 0., 0.]
        y = [0., 0., 0., 0.]

        # Get the position of the cell center
        q = self.indexToPosition(row=row, col=col)

        # Get the left top position
        x[0] = q.x - self.res_half
        y[0] = q.y - self.res_half

        # Get the right top position
        x[1] = q.x + self.res_half
        y[1] = q.y - self.res_half

        # Get the right bottom position
        x[2] = q.x + self.res_half
        y[2] = q.y + self.res_half

        # Get the left bottom position
        x[3] = q.x - self.res_half
        y[3] = q.y + self.res_half

        # Return the result
        return (x, y)

def ind2sub(n_cols: int, ind: int) -> tuple[int, int]:
    """Converts a scalar index into the row and column indices

        Inputs:
            n_cols: number of total columns
            ind: index in question

        Outputs:
            (row-index, column-index) for ind
    """
    (row_index, col_index) = np.divmod(ind, n_cols)
    return (row_index, col_index)

def sub2ind(n_cols: int, row_ind: int, col_ind: int) -> int:
    """Converts a (row_ind, col_ind) indexing into a single scalar indexing

        Inputs:
            n_cols: Number of columns in the matrix
            row_ind: Index of the row in question
            col_ind: Index of the column in question

        Outputs:
            Index of the single scalar item number
    """
    return row_ind*n_cols + col_ind

def generate_occupancy_from_polygon_world(world: PolygonWorld,
                                          res: float,
                                          x_lim: tuple[float, float],
                                          y_lim: tuple[float, float]) -> BinaryOccupancyGrid:
    """Converts an occupancy grid from a binary world

        Inputs:
            world: World to be converted to binary occupancy grid
            res: resolution of the occupancy grid
            x_lim, y_lim: Limits of the occupancy grid
    """
    # Create an occupancy grid
    grid = BinaryOccupancyGrid(res=res, x_lim=x_lim, y_lim=y_lim)

    # Loop through the grid and determine occupancy
    for row in range(grid.n_rows):
        for col in range(grid.n_cols):
            # Check to see if the position is inside an obstacle
            q = grid.indexToPosition(row=row, col=col)
            if world.inside_obstacle(q.state):
                grid.grid[row, col] = grid.OCCUPIED

            # Check to see if the cell corners are inside the obstacle
            x_vec, y_vec = grid.get_cell_box(row=row, col=col)
            for x,y in zip(x_vec, y_vec):
                if world.inside_obstacle(np.array([[x],[y]])):
                    grid.grid[row, col] = grid.OCCUPIED

    return grid

def occupancy_positions(grid: BinaryOccupancyGrid) -> tuple[list[float], # x_occupied
                                                            list[float], # y_occupied
                                                            list[float], # x_free
                                                            list[float]]:# y_free
    """Determines the occupied and free positions from an occupancy grid

        Inputs:
            grid: The occupancy grid to be evaluated

        Outputs:
            (x_occupied, y_occupied, x_free, y_free): (x,y) coordinates of occupied
            and free locations in the grid
    """
    # Initialize outputs
    x_occupied: list[float] = []
    y_occupied: list[float] = []
    x_free: list[float] = []
    y_free: list[float] = []

    # Loop through the grid to find all points
    for row in range(grid.n_rows):
        for col in range(grid.n_cols):
            # Check to see whether the position is inside/outside of an obstacle
            q = grid.indexToPosition(row=row, col=col)
            if grid.grid[row, col] == grid.OCCUPIED:
                x_occupied.append(q.x)
                y_occupied.append(q.y)
            else:
                x_free.append(q.x)
                y_free.append(q.y)

    # Return results
    return (x_occupied, y_occupied, x_free, y_free)
