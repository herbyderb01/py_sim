"""grid_planning.py: Provides a framework for visualization of planning using an occupancy grid
"""

import matplotlib.pyplot as plt
import py_sim.worlds.polygon_world as poly_world
from py_sim.path_planning.rrt_planner import StateSpace, path_smooth, rrt
from py_sim.tools.plot_constructor import create_plot_manifest
from py_sim.tools.sim_types import TwoDimArray, UnicycleState


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
    X: StateSpace = StateSpace(x_lim=x_limits, y_lim=y_limits) # space used for planning
    Xt: StateSpace = StateSpace(x_lim=(22., 22.), y_lim=(3., 3.)) # goal set

    # Create the obstacle world
    obstacle_world = poly_world.generate_world_obstacles()
    #obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create a plan
    x_start = TwoDimArray(x=-4.5, y=4.)
    x_vec, y_vec, tree = rrt(x_root=x_start,
                             X_t=Xt,
                             X=X,
                             dist=3.,
                             bias_t=100,
                             world=obstacle_world)
    x_vec_smooth, y_vec_smooth = path_smooth(x_vec=x_vec, y_vec=y_vec, world=obstacle_world)

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=y_limits,
                                 x_limits=x_limits,
                                 world=obstacle_world,
                                 graph=tree
                                 )

    # Visualize the plan
    plot_manifest.axes['Vehicle_axis'].plot(x_vec, y_vec, "-", color=(1., 0., 0., 1.), linewidth=3)
    plot_manifest.axes['Vehicle_axis'].plot(x_vec_smooth, y_vec_smooth, "-", color=(0., 1., 0., 1.), linewidth=2)
    for fig in plot_manifest.figs:
        fig.canvas.draw()
        fig.canvas.flush_events()

    # # Calculate the plan length
    # plan_length = graph.calculate_path_length(nodes=plan)
    # print('Plan length = ', plan_length)

    print('Planning finished, close figure')
    plt.show(block=True)

if __name__ == "__main__":
    test_rrt_planner()
