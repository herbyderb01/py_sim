"""grid_planning.py: Provides a framework for visualization of planning using an occupancy grid
"""

from typing import cast

import matplotlib.pyplot as plt
import numpy as np
import py_sim.path_planning.rrt_planner as rrt
import py_sim.worlds.polygon_world as poly_world
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.tools.sim_types import TwoDimArray, UnicycleState


def path_length(x_vec: list[float], y_vec: list[float]) -> float:
    """Calculates the length of the path given the x and y coordinates

        Inputs:
            x_vec: Vector of x values
            y_vec: Vector of y values

        Returns:
            Total path length
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


def test_rrt_planner() -> None:
    """ Plans a path in the following steps:
        * Create a world
        * Create a planner
        * Create the plotting
        * Incrementally calculate the plan until plan calculated
        * Calculate the plan length
    """
    # Initialize the state and control
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)
    y_limits=(-10, 15)
    x_limits=(-5, 25)
    X: rrt.StateSpace = rrt.StateSpace(x_lim=x_limits, y_lim=y_limits) # space used for planning
    Xt: rrt.StateSpace = rrt.StateSpace(x_lim=(22., 22.), y_lim=(3., 3.)) # goal set

    # Create the obstacle world
    obstacle_world = poly_world.generate_world_obstacles()
    #obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create a plan
    x_start = TwoDimArray(x=-4.5, y=4.)
    # x_vec, y_vec, tree = rrt.rrt(x_root=x_start,
    #                              X_t=Xt,
    #                              X=X,
    #                              dist=3.,
    #                              bias_t=100,
    #                              world=obstacle_world)
    # x_vec, y_vec, tree = rrt.rrt_star(x_root=x_start,
    #                                   X_t=Xt,
    #                                   X=X,
    #                                   dist=3.,
    #                                   bias_t=50,
    #                                   world=obstacle_world,
    #                                   num_iterations=10000,
    #                                   num_nearest=10)
    x_vec, y_vec, tree = rrt.rrt_star_informed(x_root=x_start,
                                               X_t=Xt,
                                               X=X,
                                               dist=3.,
                                               bias_t=50,
                                               world=obstacle_world,
                                               num_iterations=10000,
                                               num_nearest=10)

    # Smooth the resulting plan
    x_vec_smooth, y_vec_smooth = rrt.path_smooth(x_vec=x_vec, y_vec=y_vec, world=obstacle_world)

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=y_limits,
                                 x_limits=x_limits,
                                 world=obstacle_world,
                                 graph=tree,
                                 graph_node_size=0
                                 )

    # Visualize the plan
    plot_manifest.axes['Vehicle_axis'].plot(x_vec, y_vec, "-", color=(1., 0., 0., 1.), linewidth=3)
    plot_manifest.axes['Vehicle_axis'].plot(x_vec_smooth, y_vec_smooth, "-", color=(0., 1., 0., 1.), linewidth=2)
    for fig in plot_manifest.figs:
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Calculate the plan length
    plan_length_orig = path_length(x_vec=x_vec, y_vec=y_vec)
    path_length_smoothed = path_length(x_vec=x_vec_smooth, y_vec=y_vec_smooth)
    print('Plan length (original) = ', plan_length_orig, ', Path length (smoothed) = ', path_length_smoothed)

    print('Planning finished, close figure')
    plt.show(block=True)

if __name__ == "__main__":
    test_rrt_planner()
