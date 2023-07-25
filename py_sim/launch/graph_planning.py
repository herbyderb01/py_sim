"""grid_planning.py: Provides an example of path planning using a graph search with a topology graph, visibility graph, and Voronoi graph
"""

import matplotlib.pyplot as plt
import networkx as nx
import py_sim.worlds.polygon_world as poly_world
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.tools.sim_types import TwoDimArray, UnicycleState


def test_graph_planner() -> None:
    """ Plans a path using a graphical representation.

      Planning occurs in the following steps:
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

    # Create the obstacle world
    #obstacle_world = poly_world.generate_world_obstacles()
    obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create the graph to be used for planning
    #graph = poly_world.topology_world_obstacles()
    #graph = poly_world.topology_non_convex_obstacles()
    graph = poly_world.create_visibility_graph(world=obstacle_world)
    # graph = poly_world.create_voronoi_graph(world=obstacle_world, y_limits=y_limits, x_limits=x_limits, resolution=0.1)


    # Create the starting and stopping indices
    ind_start = graph.add_node_and_edges(position=TwoDimArray(x = -2., y=-3.),
                                         world=obstacle_world, n_connections=5)
    ind_end = graph.add_node_and_edges(position=TwoDimArray(x = 14., y=7.),
                                       world=obstacle_world,
                                       n_connections=5)

    # Create a plan
    plan = nx.dijkstra_path(G=graph.graph, source=ind_start, target=ind_end)

    # Visualize the plan
    x_vec, y_vec = graph.convert_to_cartesian(nodes=plan)

    # Create the manifest for the plotting
    _ = create_plot_manifest(initial_state=state_initial,
                                 y_limits=y_limits,
                                 x_limits=x_limits,
                                 world=obstacle_world,
                                 graph=graph,
                                 plan = (x_vec, y_vec)
                                 )

    # Calculate the plan length
    plan_length = graph.calculate_path_length(nodes=plan)
    print('Plan length = ', plan_length)

    print('Planning finished, close figure')
    plt.show(block=True)

if __name__ == "__main__":
    test_graph_planner()
