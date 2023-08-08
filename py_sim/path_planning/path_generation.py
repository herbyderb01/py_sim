"""create_path.py provides simple functions for creating a path through an environment
"""

from typing import Optional

import networkx as nx
import py_sim.path_planning.forward_grid_search as search
import py_sim.worlds.polygon_world as poly_world
from py_sim.sensors.occupancy_grid import generate_occupancy_from_polygon_world
from py_sim.tools.sim_types import TwoDimArray
from py_sim.worlds.polygon_world import PolygonWorld


def create_path(start: TwoDimArray,
                end: TwoDimArray,
                obstacle_world: PolygonWorld,
                plan_type: str,
                x_lim: tuple[float, float] = (-5., 25.),
                y_lim: tuple[float, float] = (-5., 10.)) -> Optional[tuple[list[float], list[float]]]:
    """ Creates a plan from the start to the end using the given world

    Args:
        start: The starting point for the plan
        end: The ending point for the plan
        obstacle_world: The world in which the planning is performed
        plan_type: identifies the planner to be used
            Grid-based planners: breadth, depth, dijkstra, astar, greedy
            Graph-based planners (both use dijkstra planning): visibility, voronoi
        x_lim: The limits on planning for the x-axis
        y_lim: The limits on planning for the y-axis

    Returns:
        tuple[list[float], list[float]]: A list of x and y coordinates for each point along the plan
    """

    # Grid-based planning
    if plan_type in ["breadth", "depth", "dijkstra", "astar", "greedy"]:
        # Create the obstacle world and occupancy grid
        grid = generate_occupancy_from_polygon_world(world=obstacle_world,
                                                     res=0.25,
                                                     x_lim=x_lim,
                                                     y_lim=y_lim)

        # Create a planner
        ind_start, _ = grid.position_to_index(q=start)
        ind_end, _ = grid.position_to_index(q=end)

        planner: search.ForwardGridSearch
        if plan_type == "breadth":
            planner = search.BreadFirstGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
        elif plan_type == "depth":
            planner = search.DepthFirstGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
        elif plan_type == "dijkstra":
            planner = search.DijkstraGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
        elif plan_type == "astar":
            planner = search.AstarGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
        elif plan_type == "greedy":
            planner = search.GreedyGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
        else:
            raise ValueError("Invalid grid-based planning type passed in")

        # Create a plan
        if not planner.search():
            raise ValueError("Invalid start and end position for planning")
        return planner.get_plan_cartesian()

    # Graph based planning
    if plan_type in ["visibility", "voronoi"]:
        # Create the graph
        if plan_type == "visibility":
            graph = poly_world.create_visibility_graph(world=obstacle_world)
        elif plan_type == "voronoi":
            graph = poly_world.create_voronoi_graph(world=obstacle_world, y_limits=y_lim, x_limits=x_lim, resolution=0.1)
        else:
            raise ValueError("Invalid graph-based planning type")

        # Create the starting and stopping indices
        ind_start = graph.add_node_and_edges(position=start,
                                             world=obstacle_world, n_connections=5)
        ind_end = graph.add_node_and_edges(position=end,
                                           world=obstacle_world,
                                           n_connections=5)

        # Create a plan
        plan = nx.dijkstra_path(G=graph.graph, source=ind_start, target=ind_end)

        # Visualize the plan
        return graph.convert_to_cartesian(nodes=plan)

    raise ValueError("plan_type provided is not defined")
