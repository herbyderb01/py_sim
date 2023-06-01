"""grid_planning.py: Provides a framework for visualization of planning using an occupancy grid
"""

import matplotlib.pyplot as plt
import networkx as nx
import py_sim.worlds.polygon_world as poly_world
from py_sim.tools.plot_constructor import create_plot_manifest
from py_sim.tools.sim_types import UnicycleState


def test_graph_planner() -> None:
    """ Plans a path in the following steps:
        * Create a world
        * Create a planner
        * Create the plotting
        * Incrementally calculate the plan until plan calculated
        * Calculate the plan length
    """
    # Initialize the state and control
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)

    # Create the obstacle world
    # obstacle_world = poly_world.generate_world_obstacles()
    obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create the graph to be used for planning
    # graph = poly_world.topology_world_obstacles()
    graph = poly_world.topology_non_convex_obstacles()

    # Create the starting and stopping indices
    ind_start = 0
    ind_end = 8

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 world=obstacle_world,
                                 graph=graph
                                 )

    # Create a plan
    plan = nx.dijkstra_path(G=graph.graph, source=ind_start, target=ind_end)

    # Visualize the plan
    x_vec, y_vec = graph.convert_to_cartesian(nodes=plan)
    plot_manifest.axes['Vehicle_axis'].plot(x_vec, y_vec, "-", color=(1., 0., 0., 1.))
    for fig in plot_manifest.figs:
        fig.canvas.draw()
        fig.canvas.flush_events()

    # Calculate the plan length
    # plan_length = planner.calculate_plan_length()
    # print('Plan length = ', plan_length)

    print('Planning finished, close figure')
    plt.show(block=True)

if __name__ == "__main__":
    test_graph_planner()
