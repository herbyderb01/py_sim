"""path_utils.py defines utility functions that are useful for working with paths
"""

from typing import cast

import numpy as np


def path_length(x_vec: list[float], y_vec: list[float]) -> float:
    """Calculates the length of the path given the x and y coordinates

    Args:
        x_vec: Vector of x values
        y_vec: Vector of y values

    Returns:
        float: Total path length
    """
    # Initialize the path variables
    path = np.array([x_vec, y_vec] )
    x_prev = path[:,0:1]
    dist: float = 0.

    # Loop through each segment of the path
    for k in range(1,len(x_vec)):
        x_next = path[:,k:k+1]
        dist += cast(float, np.linalg.norm(x_next-x_prev))
        x_prev = x_next # Setup for next iteration

    # Return the path length
    return dist
