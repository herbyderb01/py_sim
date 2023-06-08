"""rrt_planner.py defines the algorithms and proceedures needed for planning with Rapidly Exploring
   Random trees, as defined in
    "Fillet-based RRT*: A Rapid Convergence Implementation of RRT* for Curvature Constrained Vehicles"
    by James Swedeen, Greg Droge, and Randall Christensen
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from py_sim.path_planning.graph_search import DirectedPathGraph as Tree
from py_sim.path_planning.graph_search import World
from py_sim.tools.sim_types import TwoDimArray


def initialize(root: TwoDimArray) -> Tree:
    """Returns an initialized tree with the given root and no edges

        Inputs:
            root: Root node to be added to the tree
        Returns:
            Initialize tree
    """
    tree = Tree()
    tree.add_node(position=root)
    return tree

def sample(x_lim: tuple[float, float], y_lim: tuple[float, float]) -> TwoDimArray:
    """Returns a random state from the set with the specified limits

        Inputs:
            x_lim: Limits along the x-axis
            y_lim: Limits along the y-axis

        Returns:
            Position generated uniformly within the limits
    """
    # Get the random numbers for scaling in the x and y directions
    scale = np.random.random((2,))

    # Return the random position
    return TwoDimArray(x=x_lim[0]+scale.item(0)*(x_lim[1]-x_lim[0]),
                       y=y_lim[0]+scale.item(1)*(y_lim[1]-y_lim[0]))

def nearest(x: TwoDimArray, tree: Tree) -> tuple[TwoDimArray, float]:
    """Returns the state in the tree nearest to x

        Inputs:
            x: State about which the search is performed
            tree: Tree over which the search is performed

        Returns:
            x_nearest: State of the nearest node
            ind_nearest: Node index of the nearest node
    """
    # Find the nearest vertex
    points, indices = tree.nearest(point=x, n_nearest=1)

    # Extract the nearest vertex
    point = points[0]
    x_nearest = TwoDimArray(x=point.item(0), y=point.item(1))
    ind_nearest = indices[0]
    return (x_nearest, ind_nearest)

def near(x: TwoDimArray, tree: Tree, n_nearest: int) -> tuple[list[npt.NDArray[Any]], list[int]]:
    """Finds the nearest n_nearest vertices that are in the tree

        Inputs:
            x: State about which the search is performed
            tree: Tree over which the search is performed
            n_nearest: Number of nearest items to search

        Returns:
            x_nearest: List of node states in the form of (2,) vector for each state
            ind_nearest: List of node indices
    """
    return tree.nearest(point=x, n_nearest=n_nearest)

def steer(x: TwoDimArray, y: TwoDimArray, dist: float) -> TwoDimArray:
    """Returns a point that is within a predefined distance, dist, from x in the direction of y

        Inputs:
            x: originating position
            y: position defining the direction
            dist: distance along the path for the point

        Returns:
            Calculated point in the direction of y
    """
    # Create a unit vector pointing from x to y
    dist_xy = np.linalg.norm(x.state - y.state)
    if dist_xy <= dist:
        return y
    u = (y.state-x.state)/dist_xy

    # Find the point in the direction of y
    return TwoDimArray(vec=x.state+dist*u)

def insert_node(new_node: TwoDimArray, parent_ind: int, tree: Tree) -> int:
    """Adds the node, new_node, into the tree with parent_ind indicating the parent index

        Inputs:
            new_node: Node position to be added to the tree
            parent_ind: Index of the parent to which the node will be added
            tree: search tree

        Returns:
            Index of the new node within the tree
    """
    new_index = tree.add_node(position=new_node)
    tree.add_edge(node_1=parent_ind, node_2=new_index)
    return new_index

def solution(node_index: int, tree: Tree) -> tuple[list[float], list[float], list[int]]:
    """Returns the solution that runs from the tree root to the node_index through the tree

        Inputs:
            node_index: Node in question
            tree: search tree through which the path is determined

        Returns:
            x_vec: Vector of x indices
            y_vec: Vector of y indices
            ind_vec: Vector of node indices within the tree
    """
    # Initialize the solution with the node index
    position = tree.node_location[node_index]
    x_vec: list[float] = [position.item(0)]
    y_vec: list[float] = [position.item(1)]
    ind_vec: list[int] = [node_index]

    # Search through the graph from the end position to the start
    parent_itr = tree.graph.predecessors(n=node_index)
    try:
        while True:
            # Get the node information
            node_index = parent_itr.__next__()
            position = tree.node_location[node_index]

            # Store the information
            ind_vec.append(node_index)
            x_vec.append(position.item(0))
            y_vec.append(position.item(1))

            # Get the next parent info
            parent_itr = tree.node_location[node_index]
    except: # An exception is thrown at the root node as it has no parents
        pass

    # Reverse the solution so that it will go from start to end instead of end to start
    x_vec.reverse()
    y_vec.reverse()
    ind_vec.reverse()
    return (x_vec, y_vec, ind_vec)

def path(x_start: TwoDimArray, x_end: TwoDimArray) -> npt.NDArray[Any]:
    """Builds an ordered set of states that connect the state x_start to x_end without considering obstacles

        The current implementation only puts x_start and x_end together. Future modifications could
        create a series of points bewteen them

        Inputs:
            x_start: starting point
            x_end: ending point

        Returns:
            edge: 2xn matrix where each column is a state. The first column is x_start and second is x_end

    """
    edge = np.zeros((2,2))
    edge[:,0:1] = x_start.state
    edge[:,1:2] = x_start.state
    return edge

def collision_free(path: npt.NDArray[Any], world: World) -> bool:
    """Returns true if and only if path is obstacle free

        Inputs:
            path: 2xn matrix where each column is a position
            world: world used to check each individual edge for obstacles

        Returns:
            Whether (True) or not (False) the path is obstacle free
    """
    # Loop through each edge in the path
    for k in range(path.shape[1]-1): # -1 as the number of edges in a path is the number of nodes (columns) -1
        # Extract the kth edge
        edge = path[0:2,k:k+2]

        # Check for obstacle collision
        if world.intersects_obstacle(edge=edge, shrink_edge=False):
            return False # A collision was found => not collision free

    # Indicate that no collision was found
    return True
