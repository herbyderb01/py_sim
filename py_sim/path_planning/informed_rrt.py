"""informed_rrt.py defines the algorithms and proceedures needed for Informed RRT as defined in
    "Fillet-based RRT*: A Rapid Convergence Implementation of RRT* for Curvature Constrained Vehicles"
    by James Swedeen, Greg Droge, and Randall Christensen
"""

from typing import Optional

import numpy as np
import py_sim.path_planning.sampling_procedures as proc
from py_sim.path_planning.graph_search import DirectedPathGraph as Tree
from py_sim.path_planning.graph_search import World
from py_sim.path_planning.rrt_planner import extend_star, rewire
from py_sim.path_planning.sampling_procedures import Cost
from py_sim.plotting.plot_constructor import RRTPlotter
from py_sim.tools.sim_types import EllipseParameters, StateSpace, TwoDimArray


def in_ellipse(point: TwoDimArray, ell: EllipseParameters) -> bool:
    """ returns true if the given point is within the defined ellipse

    Args:
        point: The point being evaluated
        ell: Parameters defining the ellipse being evaluated

    Returns:
        bool: True if **point** in the ellipse, False otherwise
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

    Args:
        iteration: the iteration number for the sampling
        bias_t: The sampling bias period
        X: The large state space
        X_t: The smaller, target state space
        ellipse: Parameters defining the sampling ellipse

    Returns:
        TwoDimArray: Point inside the ellipse obtained through a uniform sampling distribution
    """
    # Sample until a valid point is found
    for _ in range(1000):
        x_rand = proc.biased_sample(iteration=iteration, bias_t=bias_t, X=X, X_t=X_t)
        if in_ellipse(point=x_rand, ell=ellipse):
            return x_rand
    raise ValueError("Unable to find valid sample in ellipse")

def sample_ellipse(X: StateSpace, ellipse: EllipseParameters ) -> TwoDimArray:
    """ Performs a sampling of an ellipse. If a value cannot be found then a ValueError will be raised.

    Args:
        X: The large state space
        ellipse: Parameters defining the sampling ellipse

    Returns:
        TwoDimArray: Point inside the ellipse obtained through a uniform sampling distribution
    """
    # Sample until a valid point is found
    for _ in range(1000):
        x_rand = proc.sample(X=X)
        if in_ellipse(point=x_rand, ell=ellipse):
            return x_rand
    raise ValueError("Unable to find valid sample in ellipse")

def ellipse_bounding_box(ellipse: EllipseParameters, X: StateSpace) -> StateSpace:
    """ Given ellipse parameters, a rectangular bounding box is calculated and passed out
        through the define state space

        The bounding box formula was derived through a combination from the following:
            https://www.maa.org/external_archive/joma/Volume8/Kalman/General.html
            https://www.researchgate.net/figure/Minimum-bounding-box-for-an-ellipse_fig4_327977026

    Args:
        ellipse: The parameters to define an ellipse
        X: The limiting state space (i.e., if the ellipse goes out of this state space,
            it will still form the limit for the defined bounding box)

    Returns:
        StateSpace: The resulting bounding box for the ellipse, constrained by X
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
    """InformedSampler maintains parameters required for informed sampling

    Attributes:
        _c_min(float): The lowest possible length path
        _c_best(float): The length of the best path found
        ellipse(EllipseParameters): The parameters of the ellipse for the informed search
        _X(StateSpace): Maintains the bounding state space in which to sample
        _X_bound(StateSpace): Bound for sampling ellipse

    """
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

        Args:
            iteration: the iteration number for the sampling
            bias_t: The sampling bias period
            X_t: The target state space

        Returns:
            TwoDimArray: A random sample of the informed state space
        """
        # Perform sampling as normal until the cost is found
        if self._c_best < np.inf:
            return bias_sample_ellipse(iteration=iteration,
                                   bias_t=bias_t,
                                   X=self._X_bound,
                                   X_t=X_t,
                                   ellipse=self.ellipse)
        return proc.biased_sample(iteration=iteration, bias_t=bias_t, X=self._X, X_t=X_t)

    def update_best(self, c_best: float) -> None:
        """ Updates the informed ellipse parameters based on the new best cost

        Args:
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

    Args:
        x_root: The root of the tree (i.e., the starting point of the search)
        X_t: target set
        X: state space
        dist: maximum distance for extending the tree
        bias_t: biasing of the state space
        world: the world through which the search is being made
        num_interations: Number of iterations to run the rrt_star
        num_nearest: Number of nearest agents to use in the extend-star and rewire algorithms
        plotter(Optional[RRTPlotter]): The plotter to be used during planning

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
            node_index = proc.insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

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
                              ind_goal=min_index,
                              ind_near=ind_near,
                              ind_rewire=ind_rewire,
                              sampling_ellipses=[sampler.ellipse],
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
                            sampling_ellipses=[sampler.ellipse],
                            force_plot=True)

    x_vec, y_vec, ind_vec = proc.solution(node_index=min_index, tree=tree)
    return (x_vec, y_vec, ind_vec, tree, cost)
