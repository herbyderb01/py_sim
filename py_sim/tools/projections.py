"""projections.py provides a series of functions for generating intersections and projections between line segments and points
"""

from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt


def line_segment_intersection(edge1: npt.NDArray[Any], edge2: npt.NDArray[Any]) -> Optional[npt.NDArray[Any]]:
    """ Determines the intersection point of an two line segments or None if the intersection does not exist.

        The intersection is found by determining the scaling along edge1 of the point of intersection

    Args:
        edge1: 2x2 matrix of points where each point is a column
        edge2: 2x2 matrix of points where each point is a column

    Return:
        Optional[npt.NDArray[Any]]: None if no intersection exists, the intersection point as a column vector if it does
    """
    # Extract the (x,y) coordinates of each point
    x1 = edge1[0,0] # Edge 1 start
    y1 = edge1[1,0]
    x2 = edge1[0,1] # Edge 1 end
    y2 = edge1[1,1]
    x3 = edge2[0,0] # Edge 2 start
    y3 = edge2[1,0]
    x4 = edge2[0,1] # Edge 2 end
    y4 = edge2[1,1]

    # Calculate the denominator
    denom = (y4-y3)*(x2-x1) - (x4-x3)*(y2-y1)
    if denom == 0: # parallel
        return None
    ua = ((x4-x3)*(y1-y3) - (y4-y3)*(x1-x3)) / denom
    if ua < 0 or ua > 1: # out of range
        return None
    ub = ((x2-x1)*(y1-y3) - (y2-y1)*(x1-x3)) / denom
    if ub < 0 or ub > 1: # out of range
        return None
    x = x1 + ua * (x2-x1)
    y = y1 + ua * (y2-y1)
    return np.array([[x],[y]])

def find_closest_intersection(edge1: npt.NDArray[Any],
                              edge_list: list[npt.NDArray[Any]]) -> Optional[tuple[float, npt.NDArray[Any]]]:
    """ Finds the closest intersection point between an edge and an edge list.
        Proximity is based on the location of the first point in edge 1

        Args:
            edge1: 2x2 matrix of points where each point is a column
            edge_list: list of 2x2 matrices consisting of edges

        Returns:
            Optional[tuple[float, npt.NDArray[Any]]]:
                Null if no intersection found
                (inter_dist, inter_point) if an intersection is found

                    inter_dist is the distance from the first point on edge 1 to the intersection point

                    inter_point: 2x1 matrix representing the position of intersection
    """
    # Extract the point of interest
    p1 = edge1[:,0:1]

    # Initialize the result
    inter_dist = np.inf     # Distance to the point of intersection
    inter_point = np.zeros((2,1))

    # Loop through and find the closest intersection to p1
    for edge in edge_list:
        result = line_segment_intersection(edge1=edge1, edge2=edge)
        if result is not None:
            # Calculate the distance to the result
            dist = cast(float, np.linalg.norm(p1-result))

            # Store the distance and point if better than previous
            if dist < inter_dist:
                inter_dist = dist
                inter_point = result

    # Return the result
    if np.isinf(inter_dist):
        return None
    return (inter_dist, inter_point)

def project_point_to_edge(edge: npt.NDArray[Any],
                          point: npt.NDArray[Any],
                          s_min: float = 0.,
                          s_max: float = np.inf) -> tuple[npt.NDArray[Any], float]:
    """Finds the projection of the given point onto the linesegment defined by the edge

    Args:
        edge: 2x2 matrix of points where each point is a column
        point: 2x1 matrix defining the cartesian representation of a point
        s_min: The minimum path length distance along the segment
        s_max: The maximum path length distance along the segment

    Returns:
        tuple[npt.NDArray[Any], float]: (proj, dist) The projection of the point onto the edge and the distance along the edge

            proj: Projection point along the edge bounded by the edge vertices
            dist: Distance along the edge of the projected point
    """
    # Calculate the unit vector along the edge
    a = edge[:,0:1]
    b = edge[:,1:2]
    delta = b-a
    edge_length = float(np.linalg.norm(delta))
    if edge_length <= 1.e-6:
        return (a, 0.)
    u = delta / edge_length

    # Calculate the distance along the projected line
    parallel_dist = float((point-a).transpose()@u)

    # Special case 1: s_min is larger than the edge length
    if s_min >= edge_length:
        return (b, edge_length)

    # Adjust s_min to be non-negative
    s_min = max(s_min, 0.)

    # Special case 2: projection lies before s_min (also takes into account parallel_dist < 0)
    if parallel_dist <= s_min:
        return (a + s_min*u, s_min)

    # Adjust s_max to be max of edge length
    s_max = min(s_max, edge_length)

    # Special case 3: projection lies after s_max along the segment (also accounts for projection past point b)
    if parallel_dist >= s_max:
        return (a + s_max*u, s_max)

    # Nominal case: projection lies between a and b
    return (a + parallel_dist*u, parallel_dist)

def calculate_line_distances(line: npt.NDArray[Any]) -> list[float]:
    """Calculates the distance to each point on the line

    Args:
        line: 2xn matrix defining a line where each column is a point along the line

    Returns:
        list[float]: The distance along the line to each point
    """
    # Initialize the distances with the first distance being zero
    s_val: list[float] = [0.]

    # Loop through the line and calculate the distances
    for k in range(0, line.shape[1]-1):
        # Get the points on the next line segment
        a = line[:, [k]]
        b = line[:, [k+1]]

        # Calculate the aggregate distance to point b
        s_val.append(float(np.linalg.norm(b-a))+s_val[-1])

    return s_val

def project_point_to_line(point: npt.NDArray[Any],
                          line: npt.NDArray[Any],
                          s_vals: list[float],
                          s_min: float = 0.,
                          s_max: float = np.inf) -> tuple[npt.NDArray[Any], float, int]:
    """Finds the projection of the given point onto a linesegment defined by the path bound by the minimum and maximum path distance.

    Args:
        point: 2x1 matrix defining the cartesian representation of a point
        line: 2xn matrix defining a line where each column is a point along the line
        s_vals: length n list giving the distance along the line for each point
        s_min: minimum distance allowed along the path
        s_max: maximum distance allowed along the path

    Returns:
        tuple[npt.NDArray[Any], float, int]: (proj, s_proj, k_prev)

            proj: Projection point along the line

            s_proj: The distance along the line of the projection point

            k_prev: The index of the point on the line preceding the projection point
    """
    # Ensure that the list and line values line up
    if len(s_vals) != line.shape[1]:
        raise ValueError("line and s_vals must be same length")
    if s_min > s_max:
        raise ValueError("s_min must be <= s_max")

    # Special case 1: s_min > final s_val
    if s_min >= s_vals[-1]:
        return (line[:, [-1]], s_vals[-1], len(s_vals)-1)

    # Special case 2: s_max < first s_val
    if s_max <= s_vals[0]:
        return (line[:, [0]], s_vals[0], 0)

    # Initialize the outputs
    proj = line[:,0:1] # Projection point
    s_proj = 0. # Distance along line of the projection point
    k_prev: int = 0

    # Loop through each of the line segments and find the closest point
    edge_length = 0. # Stores the length of the most recent edge
    dist_min = np.inf
    for (k, s_start) in zip(range(line.shape[1]), s_vals):
        # Extract the line segment
        edge = line[:, k:k+2]

        # Check to see if the max distance has been exceeded
        if s_start >= s_max:
            break

        # Continue if the min distance isn't met by the current iteration
        edge_length = float( np.linalg.norm(edge[:,0:1] - edge[:, 1:2]) )
        if s_start + edge_length < s_min:
            continue

        # Calculate the projection
        (proj_k, s_proj_k) = project_point_to_edge(edge=edge,
                                                   point=point,
                                                   s_min=(s_min-s_start), # Redefine s range to be along segment
                                                   s_max=(s_max-s_start))
        s_proj_k += s_start # Update the projection distance to be along the line instead of the segment
        dist_k = float( np.linalg.norm(proj_k-point) )

        # Determine if the new projection is a better projection
        if dist_k < dist_min:
            dist_min = dist_k
            s_proj = s_proj_k
            proj = proj_k
            k_prev = k


    # Return the best found projection
    return (proj, s_proj, k_prev)

def carrot_point(line: npt.NDArray[Any],
                 s_vals: list[float],
                 s_des: float,
                 k_prev: int = 0) -> npt.NDArray[Any]:
    """Get a point on the line at the distance of s_des along the line. Returns the final point if the line is shorter than s_des

    Args:
        line: 2xn matrix defining a line where each column is a point along the line
        s_vals: length n list giving the distance along the line for each point
        s_des: The point at the desired location
        k_prev: A guess of the point on the line for the distance just smaller than s_des

    Returns:
        npt.NDArray[Any]: 2x1 point along the line at s_des
    """

    # Special case: s_des before the first point
    if s_des <= s_vals[0]:
        return line[:,[0]]

    # Special case: s_des after the end of the line
    if s_des >= s_vals[-1]:
        return line[:, [-1]]

    # Find the previous point on the line
    if s_vals[k_prev] > s_des:
        k_prev = 0
    while s_vals[k_prev] < s_des: # Increments until passed s_des
        k_prev += 1
    k_prev -= 1 # Decrement by 1 to get the previous

    # Get the unit vector pointing along the segment
    a = line[:, [k_prev]]   # Extract points from the line
    b = line[:, [k_prev+1]]
    delta = float( np.linalg.norm(b-a) ) # Create the unit vector
    if delta < 1.e-8:
        raise ValueError("Line points are on top of each other")
    unit = (b-a) / delta

    # Get the desired point along the line
    dist = s_des-s_vals[k_prev]
    return cast(npt.NDArray[Any],  a + dist*unit )
