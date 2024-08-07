"""rrt_planner.py defines the algorithms and proceedures needed for planning with Rapidly Exploring
   Random trees using the rrt and rrt* planners as defined in
    "Fillet-based RRT*: A Rapid Convergence Implementation of RRT* for Curvature Constrained Vehicles"
    by James Swedeen, Greg Droge, and Randall Christensen
"""

from typing import Optional, Union, cast

import numpy as np
import py_sim.path_planning.sampling_procedures as proc
from py_sim.path_planning.graph_search import DirectedPathGraph as Tree
from py_sim.path_planning.graph_search import World
from py_sim.path_planning.sampling_procedures import Cost
from py_sim.plotting.plot_constructor import RRTPlotter
from py_sim.tools.sim_types import StateSpace, TwoDimArray


############### RRT Proceedures #############
def extend(x_rand: TwoDimArray, tree: Tree, dist: float, cost: Cost, world: World) -> tuple[TwoDimArray, int, float]:
    """ Given a sample, x_rand, and Tree, the Extend procedure finds the closest vertex to x_rand
        that is already in the tree and checks if a valid extension can be made from the tree
        towards x_rand

    Args:
        x_rand: A random (or otherwise sampled) state to be added to the tree
        tree: tree to be extended
        dist: max distance from the tree for placement of the new point
        cost: dictionary of node costs
        world: World through which the node is being evaluated

    Returns:
        tuple[TwoDimArray, int, float]:
            x_new: The new node to be added to the tree

            ind_p: The parent index for the node being added

            cost_new: The cost to come to the new point - infinite if not valid
    """
    x_nearest, ind_nearest = proc.nearest(x=x_rand, tree=tree)
    x_new = proc.steer(x=x_nearest, y=x_rand, dist=dist)
    cost_new = proc.cost_to_come(x_n=x_new, ind_p=ind_nearest, tree=tree, cost=cost, world=world)
    return x_new, ind_nearest, cost_new

def rrt(x_root: TwoDimArray,
        X_t: StateSpace,
        X: StateSpace,
        dist: float,
        bias_t: int,
        world: World,
        plotter: Optional[RRTPlotter] = None) -> \
             tuple[list[float], list[float], list[int], Tree, Cost]:
    """ Performs a search from the root node to the target set using the rapidly exploring random tree algorithm

    Args:
        x_root: The root of the tree (i.e., the starting point of the search)
        X_t: target set
        X: state space
        dist: maximum distance for extending the tree
        bias_t: biasing of the state space
        world: the world through which the search is being made
        plotter: an optional plotter for visualizing rrt

    Returns:
        tuple[list[float], list[float], list[int], Tree, Cost]:
        The path through the state space from the start to the end

            x_vec: Vector of x indices

            y_vec: Vector of y indices

            ind_vec: The indices used within tree for the solution

            tree: The resulting tree used in planning

            cost: The resulting cost for each node
    """

    # Create the tree and cost storing structures
    tree, cost = proc.initialize(root=x_root)

    # Loop through the space until a solution is found
    iteration = 0 # Stores the interation count
    while True:
        # Extend the tree towards a biased sample
        x_rand = proc.biased_sample(iteration=iteration, bias_t=bias_t, X=X, X_t=X_t)
        x_new, ind_p, cost_new = extend(x_rand=x_rand, tree=tree, dist=dist, cost=cost, world=world)

        # Check for plotting
        if plotter is not None:
            plotter.plot_plan(iteration=iteration,
                              tree=tree,
                              x_start=x_root,
                              X=X,
                              X_t=X_t,
                              x_rand=x_rand,
                              x_new=x_new,
                              ind_p= ind_p,
                              force_plot=X_t.contains(state=x_new))

        # Insert the point into the tree
        if cost_new < np.inf:
            node_index = proc.insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

            # Evaluate if the solution is complete
            if X_t.contains(state=x_new):
                x_vec, y_vec, ind_vec = proc.solution(node_index=node_index, tree=tree)
                return (x_vec, y_vec, ind_vec, tree, cost)

        # Update the interation count for the next iteration
        iteration += 1

def path_smooth(x_vec: list[float], y_vec: list[float], world: World) -> tuple[list[float], list[float]]:
    """ Smooth the set of waypoints given the world. The input path is refined in a suboptimal way
        to try to eliminate unecessary intermediary nodes

        This is an implementation of the Smooth RRT Path Algorithm 11 from Beard "Small Unmanned Aircraft" book

    Args:
        x_vec: List of x-positions
        y_vec: List of y-positions
        world: world through which the planning is occuring

    Returns:
        tuple[list[float], list[float]]:
            x_vec: smoothed x-positions

            y_vec: smoothed y-positions
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
    while j < n_nodes:
        # Create the potential edge from node i to j
        edge = np.array([[x_vec[i], x_vec[j]],
                         [y_vec[i], y_vec[j]]])

        # If the path is not feasible, then update the new path and update pointers
        if not proc.collision_free(path=edge, world=world):
            # Add point previous to j to the path (we know we can get there)
            x_vec_new.append(x_vec[j-1])
            y_vec_new.append(y_vec[j-1])

            # Update the pointers
            i = j-1

        # Update j to be the next node
        j += 1

    # Check for collision to the last node
    edge = np.array([[x_vec[i], x_vec[-1]],
                     [y_vec[i], y_vec[-1]]])
    if not proc.collision_free(path=edge, world=world):
        x_vec_new.append(x_vec[j-1])
        y_vec_new.append(y_vec[j-1])

    # Add the final node from the path
    x_vec_new.append(x_vec[-1])
    y_vec_new.append(y_vec[-1])

    # Return the new path
    return (x_vec_new, y_vec_new)


############### RRT* Proceedures ############
def extend_star(x_rand: TwoDimArray,
                tree: Tree,
                dist: float,
                cost: Cost,
                world: World,
                n_nearest: int) -> tuple[TwoDimArray, int, float, list[int]]:
    """ Given a tree, the extend* procedure finds the best "local" connection for extending the tree in the direction of
        a random sample from the state space. It returns a new point to be added to the tree as well as the parent

    Args:
        x_rand: A random (or otherwise sampled) state to be added to the tree
        tree: tree to be extended
        dist: max distance from the tree for placement of the new point
        cost: dictionary of node costs
        world: World through which the node is being evaluated
        n_nearest: Number of nearest neighbors to consider

    Returns:
        tuple[TwoDimArray, int, float, list[int]]:
            x_new: The new node to be added to the tree

            ind_p: The parent index for the node being added

            cost_new: The cost to come to the new point - infinite if not valid
    """
    # Get an extension from the tree
    x_n, ind_p, cost_n = extend(x_rand=x_rand, tree=tree, dist=dist, cost=cost, world=world)

    # Look at local neighborhood if an extension is possible
    if cost_n < np.inf:
        _, ind_near = proc.near(x=x_n, tree=tree, n_nearest=n_nearest)

        # Loop through to check for shorter path to nearest neighbors
        for ind_p_tmp in ind_near:
            # Store the data if the cost is lower
            c_tmp = proc.cost_to_come(x_n=x_n, ind_p=ind_p_tmp, tree=tree, cost=cost, world=world)
            if c_tmp < cost_n:
                ind_p = ind_p_tmp
                cost_n = c_tmp
    else:
        ind_near = []

    # Return the results
    return (x_n, ind_p, cost_n, ind_near)

def rewire(ind_p: int, ind_near: list[int], tree: Tree, cost: Cost, world: World) -> list[int]:
    """ Given a tree with node indexed by ind_p and set of neighboring nodes ind_near, rewire updates the tree
        such that ind_new is made the parent of elements in the neighboring sets if it results in a lower cost
        path for the neighbor.

    Args:
        ind_p: Index of the node about which the tree is being rewired
        ind_near: Indices of nodes that are neighbors to the node indexed by ind_new
        tree: Search tree being rewired
        cost: dictionary of node costs
        world: World through which the node is being evaluated

    Returns:
        list[int]: List of nodes to which rewiring was done through ind_p
    """

    # Loop through each of the neighboring states to evaluate the rewire
    rewired_nodes: list[int] = []
    for ind_child in ind_near:
        # Calculate the cost of going through ind_new to the potential child
        x_child = tree.get_node_position(ind_child)
        c_child = proc.cost_to_come(x_n=x_child, ind_p=ind_p, tree=tree, cost=cost, world=world)

        # Rewire if it the resulting path is lower cost
        if c_child < cost[ind_child]:
            # Indicate that the nodes have been rewired
            rewired_nodes.append(ind_child)

            # Remove the previous parent edge
            prev_parent = proc.parent(ind_c=ind_child, tree=tree)
            if prev_parent is None:
                continue
            tree.remove_edge(node_1=prev_parent, node_2=ind_child)

            # Update the parent for the child node
            tree.add_edge(node_1=ind_p, node_2=ind_child,
                          weight=proc.edge_cost(node_1=tree.get_node_position(node=ind_p),
                                           node_2=tree.get_node_position(node=ind_child)))

            # Update the cost to the child node
            cost[ind_child] = c_child

            # Update the cost to the succeeding children nodes
            children_list = proc.children(ind_p=ind_child, tree=tree)
            while len(children_list) > 0:
                # Get the child index
                ind = children_list.pop(0)

                # Update the cost to the child
                ind_parent = cast(int, proc.parent(ind_c=ind, tree=tree))
                cost[ind] = cost[ind_parent] + tree.get_edge_weight(node_1=ind_parent, node_2=ind)
    return rewired_nodes

def rrt_star(x_root: TwoDimArray,
             X_t: StateSpace,
             X: StateSpace,
             dist: float,
             bias_t: int,
             world: World,
             num_iterations: int,
             num_nearest: int,
             plotter: Optional[RRTPlotter] = None) -> \
                 tuple[list[float], list[float], list[int], Tree, Cost]:
    """ Performs a search from the root node to the target set using the rapidly exploring random tree algorithm

        Note that if X_t is a single point, the produced tree may have multiple nodes corresponding to the same goal point.

    Args:
        x_root: The root of the tree (i.e., the starting point of the search)
        X_t: target set
        X: state space
        dist: maximum distance for extending the tree
        bias_t: biasing of the state space
        world: the world through which the search is being made
        num_interations: Number of iterations to run the rrt_star
        num_nearest: Number of nearest agents to use in the extend-star and rewire algorithms
        plotter: an optional plotter for visualizing rrt

    Returns:
        tuple[list[float], list[float], list[int], Tree, Cost]:
        The path through the state space from the start to the end

            x_vec: Vector of x indices

            y_vec: Vector of y indices

            ind_vec: The indices used within tree for the solution

            tree: The resulting tree used in planning

            cost: The resulting cost for each node
    """

    # Create the tree and cost storing structures
    tree, cost = proc.initialize(root=x_root)

    # Loop through the space until a solution is found
    iteration = 0 # Stores the interation count
    min_index = -1 # Stores the index of the minimum cost terminal node
    min_cost = np.inf # Stores the cost of the shortest found solution
    for iteration in range(num_iterations):
        # Extend the tree towards a biased sample
        x_rand = proc.biased_sample(iteration=iteration, bias_t=bias_t, X=X, X_t=X_t)
        x_new, ind_p, cost_new, ind_near = \
            extend_star(x_rand=x_rand, tree=tree, dist=dist, cost=cost, world=world, n_nearest=num_nearest)

        # Insert the point into the tree
        ind_rewire: Union[list[int], None] = None
        if cost_new < np.inf:
            # Insert the new node
            node_index = proc.insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

            # Rewire the tree
            ind_rewire = rewire(ind_p=node_index, ind_near=ind_near, tree=tree, cost=cost, world=world)

            # Update the minimum cost (rewiring may make the path shorter)
            if min_index >=0:
                min_cost = cost[min_index]

            # Evaluate a newly found solution
            if X_t.contains(state=x_new):
                if cost[node_index] < min_cost:
                    min_cost = cost[node_index]
                    min_index = node_index

        # Check for plotting
        if plotter is not None:
            plotter.plot_plan(iteration=iteration,
                              tree=tree,
                              x_start=x_root,
                              X=X,
                              X_t=X_t,
                              x_rand=x_rand,
                              x_new=x_new,
                              ind_goal=min_index,
                              ind_near=ind_near,
                              ind_rewire=ind_rewire,
                              ind_p= ind_p)

        # Update the interation count for the next iteration
        iteration += 1

    # Return the best solution. If a solution has not been found, return the one that got the closest
    if min_index < 0:
        print("Solution not found, returning the point that got the closest")
        _, min_index = proc.nearest(x=proc.sample(X=X_t), tree=tree)

    # Plot the final state of the search tree
    if plotter is not None:
        plotter.plot_plan(iteration=iteration,
                            tree=tree,
                            x_start=x_root,
                            X=X,
                            X_t=X_t,
                            x_rand=x_rand,
                            x_new=x_new,
                            ind_p= ind_p,
                            ind_goal=min_index,
                            ind_near=ind_near,
                            ind_rewire=ind_rewire,
                            force_plot=True)

    x_vec, y_vec, ind_vec = proc.solution(node_index=min_index, tree=tree)
    return (x_vec, y_vec, ind_vec, tree, cost)
