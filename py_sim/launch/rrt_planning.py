"""rrt_planning.py: Provides a framework for visualization of rrt planning through an obstacle world.

Available planners include:
    * RRT - rapidly exploring random tree
    * RRT* - optimal rapidly exploring random tree
    * I-RRT* - informed RRT*. Reduces the search space through sampling of the informed ellipse.
    * S-RRT* - smart RRT*. Reduces the search space through sampling of beacons. Also smooths the path when a better path is found.
"""

from typing import Literal, Optional

import matplotlib.pyplot as plt
import py_sim.path_planning.rrt_planner as rrt
import py_sim.worlds.polygon_world as poly_world
from py_sim.path_planning.informed_rrt import rrt_star_informed
from py_sim.path_planning.smart_rrt import rrt_star_smart
from py_sim.plotting.plot_constructor import RRTPlotter, create_plot_manifest
from py_sim.tools.path_utils import path_length
from py_sim.tools.sim_types import TwoDimArray, UnicycleState


def run_rrt_planner(planner: Literal["rrt", "rrt_star", "i-rrt", "s-rrt"],
                    plot_live: bool,
                    num_iterations: int) -> None:
    """ Plans a path through the world using an RRT planner.

    The plan is created in the following steps:
        * Create a world
        * Create a planner
        * Create the plotting
        * Incrementally calculate the plan until plan calculated
        * Calculate the plan length

    Args:
        planner: Identifies the planner to be used for planning
        plot_live: Indicates whether or not the plotting should happen live. This significantly slows down the planning
        num_iterations: The number of iterations to be used in planning (used in all the rrt-star variants)
    """
    # Initialize the state and control
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)
    y_limits=(-10, 15)
    x_limits=(-5, 25)
    X: rrt.StateSpace = rrt.StateSpace(x_lim=x_limits, y_lim=y_limits) # space used for planning
    Xt: rrt.StateSpace = rrt.StateSpace(x_lim=(22., 22.), y_lim=(2., 2.)) # goal set

    # Create the obstacle world
    obstacle_world = poly_world.generate_world_obstacles()
    #obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create the RRT plotter
    plotter: Optional[RRTPlotter] = None
    if plot_live:
        plotter = RRTPlotter(world=obstacle_world, plot_iterations=22, pause_plotting=False)

    # Create a plan
    x_start = TwoDimArray(x=-4.5, y=4.)
    if planner == "rrt":
        x_vec, y_vec, _, tree, __ = rrt.rrt(x_root=x_start,
                                            X_t=Xt,
                                            X=X,
                                            dist=3.,
                                            bias_t=100,
                                            world=obstacle_world,
                                            plotter=plotter)
    elif planner == "rrt_star":
        x_vec, y_vec, _, tree, __ = rrt.rrt_star(x_root=x_start,
                                                 X_t=Xt,
                                                 X=X,
                                                 dist=3.,
                                                 bias_t=50,
                                                 world=obstacle_world,
                                                 num_iterations=num_iterations,
                                                 num_nearest=50,
                                                 plotter=plotter)
    elif planner == "i-rrt":
        x_vec, y_vec, _, tree, __ = rrt_star_informed(x_root=x_start,
                                                      X_t=Xt,
                                                      X=X,
                                                      dist=3.,
                                                      bias_t=50,
                                                      world=obstacle_world,
                                                      num_iterations=num_iterations,
                                                      num_nearest=50,
                                                      plotter=plotter)
    elif planner == "s-rrt":
        x_vec, y_vec, _, tree, __ = rrt_star_smart(x_root=x_start,
                                                   X_t=Xt,
                                                   X=X,
                                                   dist=3.,
                                                   bias_t=50,
                                                   world=obstacle_world,
                                                   num_iterations=num_iterations,
                                                   num_nearest=50,
                                                   beacon_radius=2.,
                                                   bias_explore=10,
                                                   plotter=plotter)

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
    #run_rrt_planner(planner="rrt", plot_live=False, num_iterations=1000)
    #run_rrt_planner(planner="rrt_star", plot_live=False, num_iterations=1000)
    #run_rrt_planner(planner="i-rrt", plot_live=False, num_iterations=1000)
    run_rrt_planner(planner="s-rrt", plot_live=True, num_iterations=1000)
