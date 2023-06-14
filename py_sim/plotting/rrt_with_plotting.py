""" rrt_with_plotting.py Implements the rrt algorithm with visualization updates
"""

import numpy as np
import numpy.typing as npt
import py_sim.path_planning.rrt_planner as rrt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from py_sim.path_planning.graph_search import DirectedPathGraph as Tree
from py_sim.path_planning.graph_search import World
from py_sim.tools.sim_types import TwoDimArray


def rrt_with_plotting(x_root: TwoDimArray,
                      X_t: rrt.StateSpace,
                      X: rrt.StateSpace,
                      dist: float,
                      bias_t: int,
                      world: World,
                      plot_iterations: int,
                      pause_plotting: bool) -> tuple[list[float], list[float], Tree]:
    """ Performs a search from the root node to the target set using the rapidly exploring random tree algorithm

        Inputs:
            x_root: The root of the tree (i.e., the starting point of the search)
            X_t: target set
            X: state space
            dist: maximum distance for extending the tree
            bias_t: biasing of the state space
            world: the world through which the search is being made
            plot_iterations: Every plot_iterations, the planner will be plotted
            pause_plotting: If True, each step is paused until the user enters a key

        Returns:
            The path through the state space from the start to the end
                x_vec: Vector of x indices
                y_vec: Vector of y indices
                tree: The resulting tree used in planning
    """

    # Create the tree and cost storing structures
    tree, cost = rrt.initialize(root=x_root)

    # Loop through the space until a solution is found
    iteration = 0 # Stores the interation count
    while True:
        # Extend the tree towards a biased sample
        x_rand = rrt.biased_sample(iteration=iteration, bias_t=bias_t, X=X, X_t=X_t)
        x_new, ind_p, cost_new = rrt.extend(x_rand=x_rand, tree=tree, dist=dist, cost=cost, world=world)

        # Insert the point into the tree
        if cost_new < np.inf:
            node_index = rrt.insert_node(new_node=x_new, parent_ind=ind_p, tree=tree, cost=cost)

            # Evaluate if the solution is complete
            if X_t.contains(state=x_new):
                x_vec, y_vec, _ = rrt.solution(node_index=node_index, tree=tree)
                return (x_vec, y_vec, tree)

        # # Check for plotting
        # if np.mod(iteration, plot_iterations):
        #     # Update the plot


        # Update the interation count for the next iteration
        iteration += 1