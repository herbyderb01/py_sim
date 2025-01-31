"""occupancy_grid.py Stores the classes and defs for interacting with an occupancy grid
"""

from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt
from py_sim.tools.sim_types import TwoDimArray
from py_sim.worlds.polygon_world import PolygonWorld


class BinaryOccupancyGrid:
    """Stores a binary occupancy grid

    Attributes:
        FREE(bool):  Boolean indicating that a location is free of obstacles
        OCCUPIED(bool):  Boolean indicating that a location is occupied
        res(float): Resolution of the occupancy grid
        x_lim(tuple[float, float]): min and max x-values
        y_lim(tuple[float, float]): min and max y-values
        tl_corner(TwoDimArray): Location of the top-left corner
        n_cols(int): Number of columns in the grid
        n_rows(int): Number of rows in the grid
        grid(NDArray[bool]): Binary grid storing the occupancy
        max_index(int): Stores the maximum scalar index of the grid

    """
    FREE: bool = False # Boolean indicating that a location is free of obstacles
    OCCUPIED: bool = True # Boolean indicating that a location is occupied
    def __init__(self, res: float, x_lim: tuple[float, float], y_lim: tuple[float, float]) -> None:
        """Initializes the occupance grid values

        Args:
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
        self.max_index: int = self.n_cols*(self.n_rows+1)-1 # Stores the maximum scalar index of the grid

    def is_occupied(self, q: TwoDimArray) -> bool:
        """ is_occupied returns true if the position is in an occupied
            cell, false otherwise. An invalid position is considered occupied

        Args:
            q: 2x1 position

        Returns:
            bool: False if free, true otherwise
        """

        # Calculate the indices
        (row, col, valid) = self.position_to_mat_coord(q)

        # Determine if it is occupied
        if valid:
            return cast(bool, self.grid[row, col] == self.OCCUPIED)
        return True # Default to occupied

    def inside_obstacle(self, point: npt.NDArray[Any]) -> bool:
        """ inside_obstacle returns true if the point is inside an obstacle

        Args:
            point: 2x1 point to be evaluated

        Returns:
            bool: True if the point is in an obstacle
        """
        # Convert the point to a TwoDimArray
        q = TwoDimArray(x=point.item(0), y=point.item(1))

        # Check to see if the point is occupied
        return self.is_occupied(q)

    def set_occupied(self, q: TwoDimArray) -> None:
        """ set_occupied sets the grid location corresponding to the 2x1
            position q as occupied. Does nothing if q is out of bounds

        Args:
            q: 2x1 position
        """

        # Calculate the indices
        (row, col, valid) = self.position_to_mat_coord(q)

        # Set cell as occupied
        if valid:
            self.grid[row, col] = self.OCCUPIED

    def set_free(self, q: TwoDimArray) -> None:
        """ set_free sets the grid location corresponding to the 2x1
            position q as free

        Args:
            q: 2x1 position
        """
        # Calculate the indices
        (row, col, valid) = self.position_to_mat_coord(q)

        # Set cell as Free
        if valid:
            self.grid[row, col] = self.FREE

    def position_to_mat_coord(self, q: TwoDimArray) -> tuple[int, int, bool]:
        """ position_to_mat_coord converts a position into matrix coordinates

        Args:
            q: 2x1 position corresponding to the point

        Returns:
            tuple[int, int, bool]:
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

        return (int(row), int(col), valid)

    def position_to_index(self, q: TwoDimArray) -> tuple[int, bool]:
        """ position_to_index converts a position into a scalar index into the matrix

        Args:
            q: 2x1 position corresponding to the point

        Returns:
            tuple[int, bool]:
                index: single index into the matrix

                Valid: boolean indicating whether the passed in position is valid (True)
        """
        # Convert position to matrix coordinates
        (row, col, valid) = self.position_to_mat_coord(q=q)

        # Convert matrix coordinates to scalar coordinates
        index = sub2ind(n_cols=self.n_cols, row_ind=row, col_ind=col)

        return (index, valid)

    def indices_to_position(self, row: int, col: int) -> TwoDimArray:
        """ indices_to_position converts (row,column) indices into a position

        Args:
            row: row index inside grid
            col: column index inside grid

        Returns:
            TwoDimArray: 2x1 position corresponding to the point
        """
        # Create the position
        # Negative sign comes from tl being the max y value (positive in row moves negative in y)
        q = self.tl_corner.state + np.array([[self.res*col-self.res_half],
                                             [-self.res*row+self.res_half]])

        return TwoDimArray(x=q.item(0), y=q.item(1))

    def index_to_position(self, ind: int) -> TwoDimArray:
        """ Converts a single matrix index to a position

        Args:
            ind: Matrix scalar index

        Returns:
            TwoDimArray: 2x1 position corresponding to the point
        """
        # Get the corresponding row and column
        (row, col) = ind2sub(n_cols=self.n_cols, ind=ind)

        # Get the position
        return self.indices_to_position(row=row, col=col)

    def get_cell_box(self, row: int, col: int) -> tuple[list[float],
                                                        list[float]]:
        """ Defines the bounding box for a cell given by the row and column indices

        Args:
            row: Index for the grid row
            col: Index for the grid column

        Returns:
            tuple[list[float], list[float]]:
                (x_vec, y_vec): Define the coordinates of the cell corners. Each has four elements.
        """
        # Initailize the outputs
        x = [0., 0., 0., 0.]
        y = [0., 0., 0., 0.]

        # Get the position of the cell center
        q = self.indices_to_position(row=row, col=col)

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

    def free_index(self, ind: int) -> bool:
        """Returns true if the index is valid and corresponds to an obstacle free location
        """
        # Check to see if the index is valid
        if ind < 0 or ind > self.max_index:
            return False

        # Check to see if the index points to an obstacle free location
        return bool(self.grid.item(ind) == self.FREE)

    def free_subscripts(self, row: int, col: int) -> bool:
        """Returns true if the row and column indices are valie and the resulting cell is obstacle free
        """
        # Check for validity
        if row < 0 or row >= self.n_rows or col < 0 or col >= self.n_cols:
            return False

        # Check to see if the row and column correspond to an obstacle free location
        return bool(self.grid[row, col] == self.FREE)

def ind2sub(n_cols: int, ind: int) -> tuple[int, int]:
    """Converts a scalar index into the row and column indices

    Args:
        n_cols: number of total columns
        ind: index in question

    Returns:
        tuple[int, int]:
            (row-index, column-index) for ind
    """
    (row_index, col_index) = np.divmod(ind, n_cols)
    return (int(row_index), int(col_index) )

def sub2ind(n_cols: int, row_ind: int, col_ind: int) -> int:
    """Converts a (row_ind, col_ind) indexing into a single scalar indexing

    Args:
        n_cols: Number of columns in the matrix
        row_ind: Index of the row in question
        col_ind: Index of the column in question

    Returns:
        int: Index of the single scalar item number
    """
    return int(row_ind*n_cols + col_ind)

def generate_occupancy_from_polygon_world(world: PolygonWorld,
                                          res: float,
                                          x_lim: tuple[float, float],
                                          y_lim: tuple[float, float]) -> BinaryOccupancyGrid:
    """Converts an occupancy grid from a binary world

    Args:
        world: World to be converted to binary occupancy grid
        res: resolution of the occupancy grid
        x_lim, y_lim: Limits of the occupancy grid

    Returns:
        BinaryOccupancyGrid: generated occupancy grid
    """
    # Create an occupancy grid
    grid = BinaryOccupancyGrid(res=res, x_lim=x_lim, y_lim=y_lim)

    # Loop through the grid and determine occupancy
    for row in range(grid.n_rows):
        for col in range(grid.n_cols):
            # Check to see if the position is inside an obstacle
            q = grid.indices_to_position(row=row, col=col)
            if world.inside_obstacle(q.state):
                grid.grid[row, col] = grid.OCCUPIED

            # Check to see if the cell corners are inside the obstacle
            x_vec, y_vec = grid.get_cell_box(row=row, col=col)
            for x,y in zip(x_vec, y_vec):
                if world.inside_obstacle(np.array([[x],[y]])):
                    grid.grid[row, col] = grid.OCCUPIED

    return grid

def occupancy_positions(grid: BinaryOccupancyGrid, cells: Optional[npt.NDArray[Any]]=None) \
    -> tuple[list[float], # x_occupied
       list[float], # y_occupied
       list[float], # x_free
       list[float]]:# y_free
    """Determines the occupied and free positions from an occupancy grid

        Args:
            grid: The occupancy grid to be evaluated
            cell: A binary matrix that stores the occupancy. If None, then grid.grid is used.
                  Must be same size as grid

        Returns:
            tuple[list[float], list[float], list[float], list[float]]:
                (x_occupied, y_occupied, x_free, y_free): (x,y) coordinates of occupied
                and free locations in the grid
    """
    # Extract grid
    if cells is None:
        cells = grid.grid

    # Initialize outputs
    x_occupied: list[float] = []
    y_occupied: list[float] = []
    x_free: list[float] = []
    y_free: list[float] = []

    # Loop through the grid to find all points
    for row in range(grid.n_rows):
        for col in range(grid.n_cols):
            # Check to see whether the position is inside/outside of an obstacle
            q = grid.indices_to_position(row=row, col=col)
            if cells[row, col] == grid.OCCUPIED:
                x_occupied.append(q.x)
                y_occupied.append(q.y)
            else:
                x_free.append(q.x)
                y_free.append(q.y)

    # Return results
    return (x_occupied, y_occupied, x_free, y_free)

def inflate_obstacles(grid: BinaryOccupancyGrid, inflation: float) -> BinaryOccupancyGrid:
    """Inflates the obstacles in the occupancy grid

    Args:
        grid: The occupancy grid to be inflated
        inflation: The amount of inflation to be applied

    Returns:
        BinaryOccupancyGrid: The inflated occupancy grid
    """
    # Create a new grid
    inflated_grid = BinaryOccupancyGrid(res=grid.res, x_lim=grid.x_lim, y_lim=grid.y_lim)

    # Determine the number of cells to inflate
    n_cells = int(np.ceil(inflation/grid.res))

    # Loop through the grid and determine occupancy
    for row in range(grid.n_rows): # pylint: disable=too-many-nested-blocks
        for col in range(grid.n_cols):
            # Check to see if the position is inside an obstacle
            if grid.grid[row, col] == grid.OCCUPIED:
                # Set all cells as occupied within a given radius
                for i in range(-n_cells, n_cells+1):
                    for j in range(-n_cells, n_cells+1):
                        # Check to see if the cell is within the grid
                        if row+i >= 0 and row+i < grid.n_rows and col+j >= 0 and col+j < grid.n_cols:
                            inflated_grid.grid[row+i, col+j] = inflated_grid.OCCUPIED

    return inflated_grid
