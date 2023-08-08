"""smart_rrt.py defines the algorithms and proceedures needed for Smart RRT as defined in
    "Fillet-based RRT*: A Rapid Convergence Implementation of RRT* for Curvature Constrained Vehicles"
    by James Swedeen, Greg Droge, and Randall Christensen
"""

from typing import Optional

import numpy as np
import py_sim.path_planning.sampling_procedures as proc
from py_sim.path_planning.graph_search import DirectedPathGraph as Tree
from py_sim.path_planning.graph_search import World
from py_sim.path_planning.informed_rrt import ellipse_bounding_box, sample_ellipse
from py_sim.path_planning.rrt_planner import extend_star, rewire
from py_sim.path_planning.sampling_procedures import Cost
from py_sim.plotting.plot_constructor import RRTPlotter
from py_sim.tools.sim_types import EllipseParameters, StateSpace, TwoDimArray


def optimize_path(X_b: list[int], tree: Tree, cost: Cost, world: World) -> tuple[list[float], list[float], list[int]]:
    """ Defines Algorithm 13, Optimize path. Straightens out the path to avoid watiting for
        the sampling to straigten the path.

    Args:
        X_b: a list of indices (beacons) that form a path though tree. If they do not form a path
                then this function will not optimize anything. Can be created using "solution(...)"
        tree: tree that is being created in the search
        cost: dictionary of node costs
        world: World through which the path is being evaluated

    Returns:
        tuple[list[float], list[float], list[int]]:
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

    return proc.solution(node_index=X_b[-1], tree=tree)

class SmartSampler:
    """ Defines a sampler which performs most sampling in the beacon set, sampling
        the beacons using a circle of a given radius

    Attributes:
        _X(StateSpace): Bounding state space
        _radius(float): Radius around each beacon
        _beacons(list[EllipseParameters]): The beacons (defined as ellipses)
        _bound_boxes(list[StateSpace]): The bounding box for each beacon
        _n_beacons(int): Stores the number of beacons
    """
    def __init__(self, X: StateSpace, radius: float) -> None:
        """ Initializes a smart sampler with the bounding state space and
            radius to use for each of the beacon circles

        Args:
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
        """Returns the beacons list"""
        return self._beacons

    def update_beacons(self, x_vec: list[float], y_vec: list[float]) -> None:
        """Updates the beacon locations

        Args:
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

        Args:
            iteration: the iteration number for the sampling
            bias_t: The sampling bias period
            X_t: The target state space

            Returns:
                TwoDimArray: A random, smart sampling of the state space
        """
        # Standard biased sampling if no beacons defined
        if self._n_beacons < 1:
            return proc.biased_sample(iteration=iteration, bias_t=bias_t, X=self._X, X_t=X_t)

        # Sample the state space randomly
        if np.mod(iteration, bias_explore) == 0:
            return proc.sample(X=self._X)

        # Sample the beacons
        beacon_ind = int(np.random.randint(low=0, high=self._n_beacons))
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

    Args:
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
            node_index = proc.insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

            # Rewire the tree
            ind_rewire = rewire(ind_p=node_index, ind_near=ind_near, tree=tree, cost=cost, world=world)

            # Update the minimum cost (rewiring may make the path shorter)
            if min_index >=0 and min_cost > cost[min_index]:
                min_cost = cost[min_index]
                _, __, sln_ind = proc.solution(node_index=min_index, tree=tree)
                x_vec, y_vec, _ = optimize_path(X_b=sln_ind, tree=tree, cost=cost, world=world)
                sampler.update_beacons(x_vec=x_vec, y_vec=y_vec)

            # Evaluate a newly found solution
            if X_t.contains(state=x_new):
                if cost[node_index] < min_cost:
                    min_cost = cost[node_index]
                    min_index = node_index
                    _, __, sln_ind = proc.solution(node_index=min_index, tree=tree)
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
                              ind_goal=min_index,
                              ind_near=ind_near,
                              ind_rewire=ind_rewire,
                              sampling_ellipses=sampler.beacons,
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
                          sampling_ellipses=sampler.beacons,
                          force_plot=True)

    x_vec, y_vec, ind_vec = proc.solution(node_index=min_index, tree=tree)
    return (x_vec, y_vec, ind_vec, tree, cost)
