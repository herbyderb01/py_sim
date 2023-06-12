"""rrt_planner.py defines the algorithms and proceedures needed for planning with Rapidly Exploring
   Random trees, as defined in
    "Fillet-based RRT*: A Rapid Convergence Implementation of RRT* for Curvature Constrained Vehicles"
    by James Swedeen, Greg Droge, and Randall Christensen
"""

from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt
from py_sim.path_planning.graph_search import DirectedPathGraph as Tree
from py_sim.path_planning.graph_search import World
from py_sim.tools.sim_types import TwoDimArray

Cost = dict[int, float] # Storage structure for storing the cost of a node in the graph
                        # it maps from node index to cost

class StateSpace:
    """Defines the rectangular limits of a state space"""
    def __init__(self, x_lim: tuple[float, float], y_lim: tuple[float, float]) -> None:
        """ Initializes the state space limits. Note that the limits must be increasing.

            Inputs:
                x_lim: Lower and upper limit for the x value
                y_lim: lower and upper limit for the y value
        """
        # Store the data
        self.x_lim = x_lim
        self.y_lim = y_lim

        # Check the limits
        if x_lim[1] < x_lim[0] or y_lim[1] < y_lim[0]:
            raise ValueError("Limits must be increasing")

    def contains(self, state: TwoDimArray) -> bool:
        """ Evaluates if the state is in the state space. Returns true if it is

            Inputs:
                state: State to be evaluated

            Outputs:
                True if the state is in the state space, false otherwise
        """
        return bool(state.x >= self.x_lim[0] and state.x <= self.x_lim[1] and \
                    state.y >= self.y_lim[0] and state.y <= self.y_lim[1])


################ Basic Proceedures #################
def initialize(root: TwoDimArray) -> tuple[Tree, Cost]:
    """Returns an initialized tree with the given root and no edges

        Inputs:
            root: Root node to be added to the tree
        Returns:
            Initialize tree
    """
    # Intiailize the tree
    tree = Tree()
    root_ind = tree.add_node(position=root)

    # Initialize the cost
    cost: Cost = {}
    cost[root_ind] = 0.

    return (tree, cost)

def sample(X: StateSpace) -> TwoDimArray:
    """Returns a random state from the set with the specified limits

        Inputs:
            X: State space for sampling

        Returns:
            Position generated uniformly within the limits
    """
    # Get the random numbers for scaling in the x and y directions
    scale = np.random.random((2,))

    # Return the random position
    return TwoDimArray(x=X.x_lim[0]+scale.item(0)*(X.x_lim[1]-X.x_lim[0]),
                       y=X.y_lim[0]+scale.item(1)*(X.y_lim[1]-X.y_lim[0]))

def biased_sample(iteration: int, bias_t: int, X: StateSpace, X_t: StateSpace) -> TwoDimArray:
    """ Returns a biased sampling of the state space. X is sampled uniformly except when mod(iteration, bias_t) == 0

        Inputs:
            iteration: the iteration number for the sampling
            bias_t: The sampling bias period
            X: The large state space
            X_t: The smaller, target state space

        Outputs:
            A random sample of the state space
    """
    if np.mod(iteration, bias_t) == 0:
        return sample(X=X_t)
    return sample(X=X)

def nearest(x: TwoDimArray, tree: Tree) -> tuple[TwoDimArray, int]:
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

def edge_cost(node_1: TwoDimArray, node_2: TwoDimArray) -> float:
    """Calculates the cost of an edge. Uses euclidean distance

        Inputs:
            node_1- The first point
            node_2- The second point

        Output
            The cost of the edge
    """
    diff = node_1.state - node_2.state
    return float(np.linalg.norm(diff))

def insert_node(new_node: TwoDimArray, parent_ind: int, tree: Tree, cost: Cost) -> int:
    """ Adds the node, new_node, into the tree with parent_ind indicating the parent index. The cost
        to the new node is then updated using edge_cost

        Inputs:
            new_node: Node position to be added to the tree
            parent_ind: Index of the parent to which the node will be added
            tree: search tree
            cost: dictionary of costs to be updated

        Returns:
            Index of the new node within the tree
    """
    # Calculate the edge length
    edgelength = edge_cost(node_1=new_node, node_2=tree.get_node_position(node=parent_ind))

    # Adds the index to the tree
    new_index = tree.add_node(position=new_node)
    tree.add_edge(node_1=parent_ind, node_2=new_index, weight=edgelength)

    # Adds the cost of the new node to the cost dictionary
    cost[new_index] = cost[parent_ind] + edgelength

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
            node_index_par = next(parent_itr)
            position = tree.node_location[node_index_par]

            # Store the information
            ind_vec.append(node_index_par)
            x_vec.append(position.item(0))
            y_vec.append(position.item(1))

            # Get the next parent info
            parent_itr = tree.graph.predecessors(n=node_index_par)
    except: # An exception is thrown at the root node as it has no parents
        pass

    # Reverse the solution so that it will go from start to end instead of end to start
    x_vec.reverse()
    y_vec.reverse()
    ind_vec.reverse()
    return (x_vec, y_vec, ind_vec)

def get_path(x_start: TwoDimArray, x_end: TwoDimArray) -> npt.NDArray[Any]:
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
    edge[:,1:2] = x_end.state
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

def parent(ind_c: int, tree: Tree) -> Optional[int]:
    """ Returns the parent index of the node corresponding to ind_c or None if the node is the root

        Inputs:
            ind_c: The index of the node being evaluated
            tree: search tree

        Outputs:
            Index of the parent node, None if it does not exist
    """
    try:
        parent_itr = tree.graph.predecessors(n=ind_c)
        return cast(int, next(parent_itr))
    except:
        return None

def children(ind_p: int, tree: Tree) -> list[int]:
    """ Returns every node that has ind_p as the parent index

        Inputs:
            ind_p: index of the node being evaluated
            tree: search tree

        Outputs:
            List of all of the indices to the children node of the node correponding to ind_p
    """
    children_list = cast(list[int], list(tree.graph.successors(n=ind_p)))
    return children_list

def cost_to_come(x_n: TwoDimArray, ind_p: int, tree: Tree, cost: Cost, world: World) -> float:
    """ Calculates the cost of x_n if it were connected to the tree to the parent indexed with ind_p.
    It returns infinite cost if the path is blocked

        Inputs:
            x_n: Potential new node to be added
            ind_p: The parent index to which a connection with x_n is being evaluated
            tree: search tree
            cost: dictionary of costs
            world: World through which the edge is being added

        Returns:
            cost of connecting to x_n from the indicated parent index
    """
    # Create the path from x_n to ind_p
    parent_position = tree.get_node_position(node=ind_p)
    edge = get_path(x_start=parent_position, x_end=x_n)
    if collision_free(path=edge, world=world):
        return cost[ind_p] + edge_cost(node_1=parent_position, node_2=x_n)
    return np.inf


############### RRT Proceedures #############
def extend(x_rand: TwoDimArray, tree: Tree, dist: float, cost: Cost, world: World) -> tuple[TwoDimArray, int, float]:
    """ Given a sample, x_rand, and Tree, the Extend procedure finds the closes vertex to x_rand
        that is already in the tree and checks if a valid extension can be made from the tree
        towards x_rand

        Inputs:
            x_rand: A random (or otherwise sampled) state to be added to the tree
            tree: tree to be extended
            dist: max distance from the tree for placement of the new point
            cost: dictionary of node costs
            world: World through which the node is being evaluated

        Outputs:
            x_new: The new node to be added to the tree
            ind_p: The parent index for the node being added
            cost_new: The cost to come to the new point - infinite if not valid
    """
    x_nearest, ind_nearest = nearest(x=x_rand, tree=tree)
    x_new = steer(x=x_nearest, y=x_rand, dist=dist)
    cost_new = cost_to_come(x_n=x_new, ind_p=ind_nearest, tree=tree, cost=cost, world=world)
    return x_new, ind_nearest, cost_new

def rrt(x_root: TwoDimArray,
        X_t: StateSpace,
        X: StateSpace,
        dist: float,
        bias_t: int,
        world: World) -> tuple[list[float], list[float], Tree]:
    """ Performs a search from the root node to the target set using the rapidly exploring random tree algorithm

        Inputs:
            x_root: The root of the tree (i.e., the starting point of the search)
            X_t: target set
            X: state space
            dist: maximum distance for extending the tree
            bias_t: biasing of the state space
            world: the world through which the search is being made

        Returns:
            The path through the state space from the start to the end
                x_vec: Vector of x indices
                y_vec: Vector of y indices
                tree: The resulting tree used in planning
    """

    # Create the tree and cost storing structures
    tree, cost = initialize(root=x_root)

    # Loop through the space until a solution is found
    iteration = 0 # Stores the interation count
    while True:
        # Extend the tree towards a biased sample
        x_rand = biased_sample(iteration=iteration, bias_t=bias_t, X=X, X_t=X_t)
        x_new, ind_p, cost_new = extend(x_rand=x_rand, tree=tree, dist=dist, cost=cost, world=world)

        # Insert the point into the tree
        if cost_new < np.inf:
            node_index = insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

            # Evaluate if the solution is complete
            if X_t.contains(state=x_new):
                x_vec, y_vec, _ = solution(node_index=node_index, tree=tree)
                return (x_vec, y_vec, tree)

        # Update the interation count for the next iteration
        iteration += 1

def path_smooth(x_vec: list[float], y_vec: list[float], world: World) -> tuple[list[float], list[float]]:
    """ Smooth the set of waypoints given the world. The input path is refined in a suboptimal way
        to try to eliminate unecessary intermediary nodes

        This is an implementation of the Smooth RRT Path Algorithm 11 from Beard "Small Unmanned Aircraft" book
    """
    # Check the inputs
    n_nodes = len(x_vec)
    if len(y_vec) != n_nodes or n_nodes < 2:
        raise ValueError("x and y must be same length with at least two values")

    # Initialize the outputs
    x_vec_new: list[float] = [x_vec[0]]
    y_vec_new: list[float] = [y_vec[0]]

    # Initialize the pointers
    i = 0 # Pointer to the node from which connections are being evaluated
    j = 2 # Pointer to the node to which connections are being evaluated

    # Loop through and evaluate potential edges
    while j < n_nodes-1:
        # Create the potential edge from node i to j
        edge = np.array([[x_vec[i], x_vec[j]],
                         [y_vec[i], y_vec[j]]])

        # If the path is not feasible, then update the new path and update pointers
        if not collision_free(path=edge, world=world):
            # Add point previous to j to the path (we know we can get there)
            x_vec_new.append(x_vec[j-1])
            y_vec_new.append(y_vec[j-1])

            # Update the pointers
            i = j-1

        # Update j to be the next node
        j += 1

    # Add the final node from the path
    x_vec_new.append(x_vec[-1])
    y_vec_new.append(y_vec[-1])

    # Return the new path
    return (x_vec_new, y_vec_new)