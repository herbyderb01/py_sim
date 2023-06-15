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
from py_sim.plotting.plot_constructor import RRTPlotter
from py_sim.tools.sim_types import EllipseParameters, StateSpace, TwoDimArray

Cost = dict[int, float] # Storage structure for storing the cost of a node in the graph
                        # it maps from node index to cost

################ Basic Proceedures #################
def initialize(root: TwoDimArray) -> tuple[Tree, Cost]:
    """Returns an initialized tree with the given root and no edges

        Inputs:
            root: Root node to be added to the tree
        Returns:
            Initialize tree and the resulting dictionary of costs to be updated
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
            cost: dictionary node indices to costs

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
    except StopIteration: # An exception is thrown at the root node as it has no parents
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
    except StopIteration:
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
        world: World,
        plotter: Optional[RRTPlotter] = None) -> \
             tuple[list[float], list[float], list[int], Tree, Cost]:
    """ Performs a search from the root node to the target set using the rapidly exploring random tree algorithm

        Inputs:
            x_root: The root of the tree (i.e., the starting point of the search)
            X_t: target set
            X: state space
            dist: maximum distance for extending the tree
            bias_t: biasing of the state space
            world: the world through which the search is being made
            plotter: an optional plotter for visualizing rrt

        Returns:
            The path through the state space from the start to the end
                x_vec: Vector of x indices
                y_vec: Vector of y indices
                ind_vec: The indices used within tree for the solution
                tree: The resulting tree used in planning
                cost: The resulting cost for each node
    """

    # Create the tree and cost storing structures
    tree, cost = initialize(root=x_root)

    # Loop through the space until a solution is found
    iteration = 0 # Stores the interation count
    while True:
        # Extend the tree towards a biased sample
        x_rand = biased_sample(iteration=iteration, bias_t=bias_t, X=X, X_t=X_t)
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
            node_index = insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

            # Evaluate if the solution is complete
            if X_t.contains(state=x_new):
                x_vec, y_vec, ind_vec = solution(node_index=node_index, tree=tree)
                return (x_vec, y_vec, ind_vec, tree, cost)

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
    while j < n_nodes:
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

    # Check for collision to the last node
    edge = np.array([[x_vec[i], x_vec[-1]],
                     [y_vec[i], y_vec[-1]]])
    if not collision_free(path=edge, world=world):
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

        Inputs:
            x_rand: A random (or otherwise sampled) state to be added to the tree
            tree: tree to be extended
            dist: max distance from the tree for placement of the new point
            cost: dictionary of node costs
            world: World through which the node is being evaluated
            n_nearest: Number of nearest neighbors to consider

        Outputs:
            x_new: The new node to be added to the tree
            ind_p: The parent index for the node being added
            cost_new: The cost to come to the new point - infinite if not valid
    """
    # Get an extension from the tree
    x_n, ind_p, cost_n = extend(x_rand=x_rand, tree=tree, dist=dist, cost=cost, world=world)

    # Look at local neighborhood if an extension is possible
    if cost_n < np.inf:
        _, ind_near = near(x=x_n, tree=tree, n_nearest=n_nearest)

        # Loop through to check for shorter path to nearest neighbors
        for ind_p_tmp in ind_near:
            # Store the data if the cost is lower
            c_tmp = cost_to_come(x_n=x_n, ind_p=ind_p_tmp, tree=tree, cost=cost, world=world)
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

        Inputs:
            ind_p: Index of the node about which the tree is being rewired
            ind_near: Indices of nodes that are neighbors to the node indexed by ind_new
            tree: Search tree being rewired
            cost: dictionary of node costs
            world: World through which the node is being evaluated

        Returns:
            List of nodes to which rewiring was done through ind_p
    """

    # Loop through each of the neighboring states to evaluate the rewire
    rewired_nodes: list[int] = []
    for ind_child in ind_near:
        # Calculate the cost of going through ind_new to the potential child
        x_child = tree.get_node_position(ind_child)
        c_child = cost_to_come(x_n=x_child, ind_p=ind_p, tree=tree, cost=cost, world=world)

        # Rewire if it the resulting path is lower cost
        if c_child < cost[ind_child]:
            # Indicate that the nodes have been rewired
            rewired_nodes.append(ind_child)

            # Remove the previous parent edge
            prev_parent = parent(ind_c=ind_child, tree=tree)
            if prev_parent is None:
                continue
            tree.remove_edge(node_1=prev_parent, node_2=ind_child)

            # Update the parent for the child node
            tree.add_edge(node_1=ind_p, node_2=ind_child,
                          weight=edge_cost(node_1=tree.get_node_position(node=ind_p),
                                           node_2=tree.get_node_position(node=ind_child)))

            # Update the cost to the child node
            cost[ind_child] = c_child

            # Update the cost to the succeeding children nodes
            children_list = children(ind_p=ind_child, tree=tree)
            while len(children_list) > 0:
                # Get the child index
                ind = children_list.pop(0)

                # Update the cost to the child
                ind_parent = cast(int, parent(ind_c=ind, tree=tree))
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

        Inputs:
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
            The path through the state space from the start to the end
                x_vec: Vector of x indices
                y_vec: Vector of y indices
                ind_vec: The indices used within tree for the solution
                tree: The resulting tree used in planning
                cost: The resulting cost for each node
    """

    # Create the tree and cost storing structures
    tree, cost = initialize(root=x_root)

    # Loop through the space until a solution is found
    iteration = 0 # Stores the interation count
    min_index = -1 # Stores the index of the minimum cost terminal node
    min_cost = np.inf # Stores the cost of the shortest found solution
    for iteration in range(num_iterations):
        # Extend the tree towards a biased sample
        x_rand = biased_sample(iteration=iteration, bias_t=bias_t, X=X, X_t=X_t)
        x_new, ind_p, cost_new, ind_near = \
            extend_star(x_rand=x_rand, tree=tree, dist=dist, cost=cost, world=world, n_nearest=num_nearest)

        # Insert the point into the tree
        if cost_new < np.inf:
            # Insert the new node
            node_index = insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

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
                              ind_near=ind_near,
                              ind_rewire=ind_rewire,
                              ind_p= ind_p)

        # Update the interation count for the next iteration
        iteration += 1

    # Return the best solution. If a solution has not been found, return the one that got the closest
    if min_index < 0:
        print("Solution not found, returning the point that got the closest")
        _, min_index = nearest(x=sample(X=X_t), tree=tree)

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
                            ind_near=ind_near,
                            ind_rewire=ind_rewire,
                            force_plot=True)

    x_vec, y_vec, ind_vec = solution(node_index=min_index, tree=tree)
    return (x_vec, y_vec, ind_vec, tree, cost)


############### Define I-RRT* Proceedures #####
def in_ellipse(point: TwoDimArray, ell: EllipseParameters) -> bool:
    """ returns true if the given point is within the defined ellipse

        Inputs:
            point: The point being evaluated
            ell: Parameters defining the ellipse being evaluated

        Outputs:
            True if **point** in the ellipse, False otherwise
    """
    # Translate the point so that (0,0) corresponds to the center
    p = TwoDimArray(vec=point.state-ell.center.state) # translated point so that the center is the adjusted origin

    # Evaluate the ellipse function (v == 0 <=> on boundary, v < 0 <=> inside)
    v = (p.x*ell.c_a + p.y*ell.s_a)**2 / ell.a**2 + (p.x*ell.s_a-p.y*ell.c_a)**2/ell.b**2 - 1

    return bool(v < 0)

def bias_sample_ellipse(iteration: int, bias_t: int, X: StateSpace, X_t: StateSpace,
                        ellipse: EllipseParameters ) -> TwoDimArray:
    """ Performs a biased sampling of an ellipse. Note that X_t in X X_t must intersect with the
        ellipse. If a value cannot be found then a ValueError will be raised

        Inputs:
            iteration: the iteration number for the sampling
            bias_t: The sampling bias period
            X: The large state space
            X_t: The smaller, target state space
            ellipse: Parameters defining the sampling ellipse

        Returns:
            Point inside the ellipse obtained through a uniform sampling distribution
    """
    # Sample until a valid point is found
    for _ in range(1000):
        x_rand = biased_sample(iteration=iteration, bias_t=bias_t, X=X, X_t=X_t)
        if in_ellipse(point=x_rand, ell=ellipse):
            return x_rand
    raise ValueError("Unable to find valid sample in ellipse")

def sample_ellipse(X: StateSpace, ellipse: EllipseParameters ) -> TwoDimArray:
    """ Performs a sampling of an ellipse. If a value cannot be found then a ValueError will be raised.

        Inputs:
            X: The large state space
            ellipse: Parameters defining the sampling ellipse

        Returns:
            Point inside the ellipse obtained through a uniform sampling distribution
    """
    # Sample until a valid point is found
    for _ in range(1000):
        x_rand = sample(X=X)
        if in_ellipse(point=x_rand, ell=ellipse):
            return x_rand
    raise ValueError("Unable to find valid sample in ellipse")

def ellipse_bounding_box(ellipse: EllipseParameters, X: StateSpace) -> StateSpace:
    """ Given ellipse parameters, a rectangular bounding box is calculated and passed out
        through the define state space

        The bounding box formula was derived through a combination from the following:
            https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
            https://www.researchgate.net/figure/Minimum-bounding-box-for-an-ellipse_fig4_327977026

        Inputs:
            ellipse: The parameters to define an ellipse
            X: The limiting state space (i.e., if the ellipse goes out of this state space, it will still form the limit for the defined bounding box)

        Returns:
            The resulting bounding box for the ellipse, constrained by X
    """
    # Determine range of ellipse bounding box
    A = ellipse.c_a**2/ellipse.a**2 + ellipse.s_a**2/ellipse.b**2
    B = 2*ellipse.c_a*ellipse.s_a*(1/ellipse.a**2 - 1/ellipse.b**2)
    C = ellipse.s_a**2/ellipse.a**2+ellipse.c_a**2/ellipse.b**2
    F = -1.
    x_diff = np.sqrt((4.*C*F)/(B**2-4*A*C))
    y_diff =np.sqrt((4.*A*F)/(B**2-4*A*C))

    # Update ellipse bounding box parameters
    x_lim_low = np.max([X.x_lim[0], ellipse.center.x-x_diff])
    x_lim_up = np.min([X.x_lim[1], ellipse.center.x+x_diff])
    y_lim_low = np.max([X.y_lim[0], ellipse.center.y-y_diff])
    y_lim_up = np.min([X.y_lim[1], ellipse.center.y+y_diff])
    return StateSpace(x_lim=(x_lim_low, x_lim_up), y_lim=(y_lim_low, y_lim_up))

class InformedSampler:
    """InformedSampler maintains parameters required for informed sampling"""
    def __init__(self, X: StateSpace, x_start: TwoDimArray, x_end: TwoDimArray) -> None:
        """Create the initial parameters"""
        self._c_min = np.linalg.norm(x_end.state-x_start.state)
        self._c_best: float = np.inf # The best cost seen / major axis diameter

        # Ellipse parameters
        self.ellipse = EllipseParameters(a=np.inf, b = np.inf,
                        center = TwoDimArray(vec=(x_start.state+x_end.state)/2.),
                        alpha = np.arctan2(x_end.y-x_start.y, x_end.x-x_start.x) )

        # Bounding box parameters
        self._X = X # Maintains the bounding state space in which to sample
        self._X_bound = StateSpace(x_lim=self._X.x_lim, y_lim=self._X.y_lim) # Bound for sampling ellipse

    def sample(self, iteration: int, bias_t: int, X_t: StateSpace) -> TwoDimArray:
        """Performs a biased sampling over the informed ellipse

            Inputs:
                iteration: the iteration number for the sampling
                bias_t: The sampling bias period
                X_t: The target state space

            Returns:
                A random sample of the informed state space
        """
        # Perform sampling as normal until the cost is found
        if self._c_best < np.inf:
            return bias_sample_ellipse(iteration=iteration,
                                   bias_t=bias_t,
                                   X=self._X_bound,
                                   X_t=X_t,
                                   ellipse=self.ellipse)
        return biased_sample(iteration=iteration, bias_t=bias_t, X=self._X, X_t=X_t)


    def update_best(self, c_best: float) -> None:
        """ Updates the informed ellipse parameters based on the new best cost

            Inputs:
                c_best: The best path cost seen
        """
        # Update ellipse parameters
        self._c_best = c_best
        self.ellipse.a = c_best/2.
        self.ellipse.b = np.sqrt(self._c_best**2 - self._c_min**2) / 2.

        # Update ellipse bounding box parameters
        self._X_bound = ellipse_bounding_box(ellipse=self.ellipse, X=self._X)

def rrt_star_informed(x_root: TwoDimArray,
                      X_t: StateSpace,
                      X: StateSpace,
                      dist: float,
                      bias_t: int,
                      world: World,
                      num_iterations: int,
                      num_nearest: int,
                      plotter: Optional[RRTPlotter] = None) -> \
                        tuple[list[float], list[float], list[int], Tree, Cost]:
    """ Performs a search from the root node to the target set using the rapidly exploring
        random tree algorithm with an informed sampling set

        Note that if X_t is a single point, the produced tree may have multiple nodes corresponding to the same goal point.

        Inputs:
            x_root: The root of the tree (i.e., the starting point of the search)
            X_t: target set
            X: state space
            dist: maximum distance for extending the tree
            bias_t: biasing of the state space
            world: the world through which the search is being made
            num_interations: Number of iterations to run the rrt_star
            num_nearest: Number of nearest agents to use in the extend-star and rewire algorithms

        Returns:
            The path through the state space from the start to the end
                x_vec: Vector of x indices
                y_vec: Vector of y indices
                ind_vec: The indices used within tree for the solution
                tree: The resulting tree used in planning
                cost: The resulting cost for each node

    """

    # Create the tree and cost storing structures
    tree, cost = initialize(root=x_root)

    # Create the sampler
    sampler = InformedSampler(X=X, x_start=x_root, x_end=X_t.furthest_point(x=x_root))

    # Loop through the space until a solution is found
    iteration = 0 # Stores the interation count
    min_index = -1 # Stores the index of the minimum cost terminal node
    min_cost = np.inf # Stores the cost of the shortest found solution
    for iteration in range(num_iterations):
        # Extend the tree towards a biased sample
        x_rand = sampler.sample(iteration=iteration, bias_t=bias_t, X_t=X_t)
        x_new, ind_p, cost_new, ind_near = \
            extend_star(x_rand=x_rand, tree=tree, dist=dist, cost=cost, world=world, n_nearest=num_nearest)

        # Insert the point into the tree
        if cost_new < np.inf:
            # Insert the new node
            node_index = insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

            # Rewire the tree
            ind_rewire = rewire(ind_p=node_index, ind_near=ind_near, tree=tree, cost=cost, world=world)

            # Update the minimum cost (rewiring may make the path shorter)
            if min_index >=0 and min_cost > cost[min_index]:
                min_cost = cost[min_index]
                sampler.update_best(c_best=min_cost)

            # Evaluate a newly found solution
            if X_t.contains(state=x_new):
                if cost[node_index] < min_cost:
                    min_cost = cost[node_index]
                    min_index = node_index
                    sampler.update_best(c_best=min_cost)

        # Check for plotting
        if plotter is not None:
            plotter.plot_plan(iteration=iteration,
                              tree=tree,
                              x_start=x_root,
                              X=X,
                              X_t=X_t,
                              x_rand=x_rand,
                              x_new=x_new,
                              ind_near=ind_near,
                              ind_rewire=ind_rewire,
                              sampling_ellipses=[sampler.ellipse],
                              ind_p= ind_p)

        # Update the interation count for the next iteration
        iteration += 1

    # Return the best solution. If a solution has not been found, return the one that got the closest
    if min_index < 0:
        print("Solution not found, returning the point that got the closest")
        _, min_index = nearest(x=sample(X=X_t), tree=tree)

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
                            ind_near=ind_near,
                            ind_rewire=ind_rewire,
                            sampling_ellipses=[sampler.ellipse],
                            force_plot=True)

    x_vec, y_vec, ind_vec = solution(node_index=min_index, tree=tree)
    return (x_vec, y_vec, ind_vec, tree, cost)

############### Define S-RRT* Proceedures ######
def optimize_path(X_b: list[int], tree: Tree, cost: Cost, world: World) -> tuple[list[float], list[float], list[int]]:
    """ Defines Algorithm 13, Optimize path. Straightens out the path to avoid watiting for
        the sampling to straigten the path.

        Inputs:
            X_b: a list of indices (beacons) that form a path though tree. If they do not form a path
                  then this function will not optimize anything. Can be created using "solution(...)"
            tree: tree that is being created in the search
            cost: dictionary of node costs
            world: World through which the path is being evaluated

        Returns:
            a revised solution (x, y, index) vectors without unnecessary intermediary nodes
    """
    # Setup indices for evaluating the paths
    p = 0       # The index to the parent in the path
    p_next = 1  # The index to the next parent to evaluate in the path
    n_nodes = len(X_b)
    while p < n_nodes:

        # Evaluate to all possible children
        for c_poss in range(p+1, n_nodes):
            # Attempt to rewire p to c_poss
            res = rewire(ind_p=X_b[p], ind_near=[X_b[c_poss]], tree=tree, cost=cost, world=world)

            # If rewire successful, then skep all nodes between p and c_poss when updating p
            if X_b[c_poss] in res:
                p_next = c_poss

        # Update p for the next iteration
        p = p_next
        p_next = p + 1

    return solution(node_index=X_b[-1], tree=tree)

class SmartSampler:
    """ Defines a sampler which performs most sampling in the beacon set, sampling
        the beacons using a circle of a given radius
    """
    def __init__(self, X: StateSpace, radius: float) -> None:
        """ Initializes a smart sampler with the bounding state space and
            radius to use for each of the beacon circles

            Inputs:
                X: The bounding state space
                radius: The radius of each of the circles to use around the beacons
        """
        # Store inputs
        self._X = X                                 # Bounding state space
        self._radius = radius                       # Radius around each beacon
        self._beacons: list[EllipseParameters] = [] # The beacons (defined as ellipses)
        self._bound_boxes: list[StateSpace] = []    # The bounding box for each beacon
        self._n_beacons: int = 0                    # Stores the number of beacons

    @property
    def beacons(self) -> list[EllipseParameters]:
        """Returns the beacons"""
        return self._beacons

    def update_beacons(self, x_vec: list[float], y_vec: list[float]) -> None:
        """Updates the beacon locations

            Inputs:
                x_vec: The x-coordinate for the beacon positions
                y_vec: The y-coordinate for the beacon positions
        """
        # Loop through and create new beaconds and bounding boxes
        self._beacons = []
        self._bound_boxes = []
        for (x,y) in zip(x_vec, y_vec):
            # Create the ellipse parameters
            ellipse = EllipseParameters(a=self._radius, b=self._radius, alpha=0.,
                                        center=TwoDimArray(x=x, y=y))

            # Store the ellipse and bounding box
            self._beacons.append(ellipse)
            self._bound_boxes.append(ellipse_bounding_box(ellipse=ellipse, X=self._X))

        # Update the number of beacons
        self._n_beacons = len(self._beacons)

    def sample(self, iteration: int, bias_t: int, X_t: StateSpace, bias_explore: int) -> TwoDimArray:
        """ Uses smart sampling sample the state space. Before the first beacons are initialize, the
            full state space is sampled using biased sampling. Once the beacons are establish, the
            sampling is performed nearly exclusively in the beacon set with every bias_explore
            samples being in the state space

            Inputs:
                iteration: the iteration number for the sampling
                bias_t: The sampling bias period
                X_t: The target state space

            Returns:
                A random, smart sampling of the state space
        """
        # Standard biased sampling if no beacons defined
        if self._n_beacons < 1:
            return biased_sample(iteration=iteration, bias_t=bias_t, X=self._X, X_t=X_t)

        # Sample the state space randomly
        if np.mod(iteration, bias_explore) == 0:
            return sample(X=self._X)

        # Sample the beacons
        beacon_ind = np.random.randint(low=0, high=self._n_beacons)
        return sample_ellipse(X=self._bound_boxes[beacon_ind],
                              ellipse=self._beacons[beacon_ind])

def rrt_star_smart(x_root: TwoDimArray,
                   X_t: StateSpace,
                   X: StateSpace,
                   dist: float,
                   bias_t: int,
                   world: World,
                   num_iterations: int,
                   num_nearest: int,
                   beacon_radius: float,
                   bias_explore: int,
                   plotter: Optional[RRTPlotter] = None) -> \
                        tuple[list[float], list[float], list[int], Tree, Cost]:
    """ Performs a search from the root node to the target set using the rapidly exploring
        random tree algorithm with a smart sampling set

        Note that if X_t is a single point, the produced tree may have multiple nodes corresponding to the same goal point.

        Inputs:
            x_root: The root of the tree (i.e., the starting point of the search)
            X_t: target set
            X: state space
            dist: maximum distance for extending the tree
            bias_t: biasing of the state space
            world: the world through which the search is being made
            num_interations: Number of iterations to run the rrt_star
            num_nearest: Number of nearest agents to use in the extend-star and rewire algorithms
            beacon_radius: The radius of each beacon to search
            bias_explore: The bias for exploring instead of searching within beacon

        Returns:
            The path through the state space from the start to the end
                x_vec: Vector of x indices
                y_vec: Vector of y indices
                ind_vec: The indices used within tree for the solution
                tree: The resulting tree used in planning
                cost: The resulting cost for each node

    """

    # Create the tree and cost storing structures
    tree, cost = initialize(root=x_root)

    # Create the sampler
    sampler = SmartSampler(X=X, radius=beacon_radius)

    # Loop through the space until a solution is found
    iteration = 0 # Stores the interation count
    min_index = -1 # Stores the index of the minimum cost terminal node
    min_cost = np.inf # Stores the cost of the shortest found solution
    for iteration in range(num_iterations):
        # Extend the tree towards a biased sample
        x_rand = sampler.sample(iteration=iteration, bias_t=bias_t, X_t=X_t, bias_explore=bias_explore)
        x_new, ind_p, cost_new, ind_near = \
            extend_star(x_rand=x_rand, tree=tree, dist=dist, cost=cost, world=world, n_nearest=num_nearest)

        # Insert the point into the tree
        if cost_new < np.inf:
            # Insert the new node
            node_index = insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

            # Rewire the tree
            ind_rewire = rewire(ind_p=node_index, ind_near=ind_near, tree=tree, cost=cost, world=world)

            # Update the minimum cost (rewiring may make the path shorter)
            if min_index >=0 and min_cost > cost[min_index]:
                min_cost = cost[min_index]
                _, __, sln_ind = solution(node_index=min_index, tree=tree)
                x_vec, y_vec, _ = optimize_path(X_b=sln_ind, tree=tree, cost=cost, world=world)
                sampler.update_beacons(x_vec=x_vec, y_vec=y_vec)

            # Evaluate a newly found solution
            if X_t.contains(state=x_new):
                if cost[node_index] < min_cost:
                    min_cost = cost[node_index]
                    min_index = node_index
                    _, __, sln_ind = solution(node_index=min_index, tree=tree)
                    x_vec, y_vec, _ = optimize_path(X_b=sln_ind, tree=tree, cost=cost, world=world)
                    sampler.update_beacons(x_vec=x_vec, y_vec=y_vec)

        # Check for plotting
        if plotter is not None:
            plotter.plot_plan(iteration=iteration,
                              tree=tree,
                              x_start=x_root,
                              X=X,
                              X_t=X_t,
                              x_rand=x_rand,
                              x_new=x_new,
                              ind_near=ind_near,
                              ind_rewire=ind_rewire,
                              sampling_ellipses=sampler.beacons,
                              ind_p= ind_p)

        # Update the interation count for the next iteration
        iteration += 1

    # Return the best solution. If a solution has not been found, return the one that got the closest
    if min_index < 0:
        print("Solution not found, returning the point that got the closest")
        _, min_index = nearest(x=sample(X=X_t), tree=tree)

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
                          ind_near=ind_near,
                          ind_rewire=ind_rewire,
                          sampling_ellipses=sampler.beacons,
                          force_plot=True)

    x_vec, y_vec, ind_vec = solution(node_index=min_index, tree=tree)
    return (x_vec, y_vec, ind_vec, tree, cost)
