"""plot_constructor.py: Constructs plot manifiests
"""
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import py_sim.tools.plotting as pt
from py_sim.path_planning.forward_grid_search import ForwardGridSearch
from py_sim.path_planning.graph_search import PathGraph
from py_sim.sensors.occupancy_grid import BinaryOccupancyGrid
from py_sim.tools.sim_types import UnicycleStateType, VectorField
from py_sim.worlds.polygon_world import PolygonWorld


def create_plot_manifest(initial_state: UnicycleStateType, # pylint: disable=too-many-arguments
                         y_limits: tuple[float, float],
                         x_limits: tuple[float, float],
                         position_dot: bool = False,
                         position_triangle: bool = False,
                         state_trajectory: bool = False,
                         unicycle_time_series: bool = False,
                         color: pt.Color = (0.2, 0.36, 0.78, 1.0),
                         vectorfield: Optional[VectorField] = None,
                         vector_res: float = 0.25,
                         world: Optional[PolygonWorld] = None,
                         grid: Optional[BinaryOccupancyGrid] = None,
                         plot_occupancy_grid: bool = False,
                         plot_occupancy_cells: bool = False,
                         plot_occupancy_circles: bool = False,
                         range_bearing_locations: bool = False,
                         range_bearing_lines: bool = False,
                         planner: Optional[ForwardGridSearch] = None,
                         graph: Optional[PathGraph] = None
                         ) -> pt.PlotManifest[UnicycleStateType]:
    """Creates a plot manifest given the following inputs

        Inputs:
            y_limits: min and max y-values to plot
            x_limits: min and max x-values to plot
            position_dot: Plot the vehicle position as a circle
            position_triangle: Plot the vehicle position as a triangle
            state_trajectory: Plot the trajectory that the vehicle travelled
            unicycle_time_series: Plot the time series state
            color: Color of the state plots
            vectorfield: The vectorfield class to be plotted
            vector_res: The grid resolution of the vectorfield
            world: Polygon world in operation
            grid: Occupancy grid to be used for plotting,
            plot_occupancy_grid: True -> a grid will be plotted displaying occupancy,
                                 (fairly fast and very accurate)
            plot_occupancy_cells: True -> a square for each occupied cell will be plotted
                                  (slow, but accurate),
            plot_occupancy_circles: True -> a circle will be plotted in each occupied cell
                                    (fast, but approximate),
            range_bearing_locations: plot the range measurement locations
            range_bearing_lines: plot the lines for the range bearing measurements
            planner: plots the occupancy grid, visited nodes, and planned path of the grid planner
            graph: plots the graph of possible paths
    """
    # Create the manifest to be returned
    plots = pt.PlotManifest[UnicycleStateType]()

    # Create the desired data plots
    if unicycle_time_series:
        state_plot = pt.UnicycleTimeSeriesPlot[UnicycleStateType](color=color)
        plots.data_plots.append(state_plot)
        plots.figs.append(state_plot.fig)

    # Initialize the plotting of the vehicle visualization
    fig, ax = plt.subplots()
    plots.figs.append(fig)
    plots.axes['Vehicle_axis'] = ax
    ax.set_title("Vehicle plot")
    ax.set_ylim(ymin=y_limits[0], ymax=y_limits[1])
    ax.set_xlim(xmin=x_limits[0], xmax=x_limits[1])
    ax.set_aspect('equal', 'box')

    if grid is not None:
        if plot_occupancy_grid:
            pt.plot_occupancy_grid_image(ax=ax, grid=grid)

        if plot_occupancy_cells:
            pt.plot_occupancy_grid_cells(ax=ax, grid=grid)

        if plot_occupancy_circles:
            pt.plot_occupancy_grid_circles(ax=ax, grid=grid, color_occupied=pt.red)

    if planner is not None:
        plots.data_plots.append(pt.PlanVisitedGridPlotter(ax=ax, planner=planner) )


    # Plot the world
    if world is not None:
        pt.plot_polygon_world(ax=ax, world=world)

    # Create the desired state plots
    if position_dot:
        plots.state_plots.append(pt.PositionPlot(ax=ax, label="Vehicle", color=color) )
    if position_triangle:
        plots.state_plots.append(pt.PosePlot(ax=ax, rad=0.2))

    # Create the state trajectory plot
    if state_trajectory:
        plots.data_plots.append(pt.StateTrajPlot(ax=ax, label="Vehicle Trajectory", \
                                color=color, location=initial_state))

    # Create a vectorfield plot
    if vectorfield is not None:
        plots.data_plots.append(
            pt.VectorFieldPlot(ax=ax,
                               color=color,
                               y_limits=y_limits,
                               x_limits=x_limits,
                               resolution=vector_res,
                               vector_field=vectorfield))

    # Range and bearing sensor plots
    if range_bearing_locations:
        plots.data_plots.append(pt.RangeBearingPlot(ax=ax))
    if range_bearing_lines:
        plots.data_plots.append(pt.RangeBearingLines(ax=ax))

    # Forward planner plots
    if planner is not None:
        # Plot the visited nodes (optional - replaced by PlanVisitedGridPlotter)
        # plots.data_plots.append(
        #     pt.PlanVisitedPlotter(ax=ax, planner=planner)
        # )

        # Plot the nodes in queue
        plots.data_plots.append(
            pt.PlanQueuePlotter(ax=ax, planner=planner)
        )

        # Plot the plan
        plots.data_plots.append(
            pt.PlanPlotter(ax=ax, planner=planner, ind_start=planner.ind_start, ind_end=planner.ind_end)
        )

    # Graph plot
    if graph is not None:
        nx.drawing.nx_pylab.draw(G=graph.graph, pos=graph.node_location, ax=ax )

    return plots
