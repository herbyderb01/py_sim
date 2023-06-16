"""plot_constructor.py: Constructs plot manifiests
"""
from typing import Optional

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import py_sim.plotting.plotting as pt
from matplotlib.figure import Figure
from matplotlib.patches import Ellipse
from py_sim.path_planning.forward_grid_search import ForwardGridSearch
from py_sim.path_planning.graph_search import DirectedPathGraph as Tree
from py_sim.path_planning.graph_search import GraphType, PathGraph
from py_sim.path_planning.rrt_procedures import solution
from py_sim.sensors.occupancy_grid import BinaryOccupancyGrid
from py_sim.tools.sim_types import (
    EllipseParameters,
    StateSpace,
    TwoDimArray,
    UnicycleStateType,
    VectorField,
)
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
                         graph: Optional[PathGraph[GraphType]] = None,
                         graph_node_size: int = 10
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
            graph_node_size: The size of the node circle in the graph plot
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
        nx.drawing.nx_pylab.draw(G=graph.graph, pos=graph.node_location, ax=ax, node_size=graph_node_size )

    return plots

def plot_rrt(pause_plotting: bool,
             world: PolygonWorld,
             tree: Tree,
             x_start: TwoDimArray,
             X: StateSpace,
             X_t: StateSpace,
             x_rand: TwoDimArray,
             x_new: TwoDimArray,
             ind_p: int,
             ind_goal: int = -1,
             ind_near: Optional[list[int]] = None,
             ind_rewire: Optional[list[int]] = None,
             sampling_ellipses: Optional[list[EllipseParameters]] = None,
             fig: Optional[Figure] = None) -> Figure:
    """Plots the rrt data

        Inputs:
            pause_plotting: If True, each step is paused until the user enters a key
            world: The polygon world in which the plotting occurs
            tree: Underlying tree for the plan
            x_start: the start location of the plan
            X: state space
            X_t: target set
            x_rand: The newly sampled point
            x_new: The point that the planner is attempting to add to the tree
            ind_p: The parent index within tree to which x_new it being added
            ind_goal: The index to the lowest cost goal location within the tree
            ind_near: The set of nearest neighbors over which the search is performed
            ind_rewire: The set of nodes that were rewired through x_new
            sampling_ellipses: List of all ellipses used for sampling
            fig: the figure on which to make the plots

        Returns:
            The figure on which the data is plotted
    """
    # Initialize the figure
    if fig is None:
        fig, ax = plt.subplots()
    else:
        fig.clear()
        ax = fig.add_axes(rect=(0, 0, 1, 1))

    # Setup the axes
    ax.set_title("RRT planner plot")
    ax.set_ylim(ymin=X.y_lim[0], ymax=X.y_lim[1])
    ax.set_xlim(xmin=X.x_lim[0], xmax=X.x_lim[1])

    # Plot the sampling ellipses
    if sampling_ellipses is not None:
        for ell in sampling_ellipses:
            ax.add_patch(p=Ellipse(xy=(ell.center.x, ell.center.y), width=2.*ell.a, height=2.*ell.b,
                                   angle=np.rad2deg(ell.alpha), visible=True, fill=True,
                                   facecolor=(0., 0., 1., 0.25)))

    # Plot the world
    pt.plot_polygon_world(ax=ax, world=world)

    # Plot the graph
    nx.drawing.nx_pylab.draw(G=tree.graph, pos=tree.node_location, ax=ax, node_size=0.1 )

    # Plot the randomly chosen location
    pt.initialize_position_plot(ax=ax, color=(0., 1., 1., 1.), location=x_rand)

    # Plot the new and parent positions as well as the new possible edge
    pt.initialize_position_plot(ax=ax, color=(1., 0., 1., 1.), location=x_new)
    x_parent = tree.get_node_position(node=ind_p)
    pt.initialize_position_plot(ax=ax, color=(0., 1., 0., 1.), location=x_parent)
    ax.plot([x_parent.x, x_new.x], [x_parent.y, x_new.y], color=(0., 1., 0., 1.))

    # Loop through and plot all of the neighborhood set
    if ind_near is not None:
        for ind in ind_near:
            pt.initialize_position_plot(ax=ax,
                                        color=(0., 0., 0., 1.),
                                        location=tree.get_node_position(node=ind))

    # Plot the rewire set
    if ind_rewire is not None:
        for ind in ind_rewire:
            x_neigh = tree.get_node_position(node=ind)
            ax.plot([x_new.x, x_neigh.x], [x_new.y, x_neigh.y], color=(0., 0., 1., 1.))

    # Plot the start and goal locations
    pt.initialize_position_plot(ax=ax, color=(1., 0., 0., 1.), location=x_start)
    pt.initialize_position_plot(ax=ax, color=(1., 0., 0., 1.), location=TwoDimArray(x=X_t.x_lim[0], y=X_t.y_lim[0]))

    # Plot the path to the goal
    if ind_goal >= 0:
        x_vec, y_vec, _ = solution(node_index=ind_goal, tree=tree)
        ax.plot(x_vec, y_vec, color=(1., 0., 0., 1.))

    # Get the figure ready for plotting
    fig.canvas.draw()
    fig.canvas.flush_events()
    plt.show(block=False)

    # Stop if requested
    if pause_plotting:
        input("press any key to continue")

    return fig

class RRTPlotter:
    """Stores data needed for plotting rrt plans
    """
    def __init__(self, world: PolygonWorld, plot_iterations: int, pause_plotting: bool) -> None:
        """ Initializes the RRT Plotter with the data needed for plotting the rrt plan

            Inputs:
                world: The polygon world in which the plotting occurs
                plot_iterations: Every plot_iterations, the planner will be plotted
                pause_plotting: If True, each step is paused until the user enters a key
        """
        self.world = world                      # Polygon world to draw
        self.plot_iterations = plot_iterations  # Every plot_iterations, the planner will be plotted
        self.pause_plotting = pause_plotting    # If True, each step is paused until the user enters a key
        self.fig: Optional[Figure] = None       # Figure on which the plotting occurs

    def plot_plan(self,
                  iteration: int,
                  tree: Tree,
                  x_start: TwoDimArray,
                  X: StateSpace,
                  X_t: StateSpace,
                  x_rand: TwoDimArray,
                  x_new: TwoDimArray,
                  ind_p: int,
                  ind_goal: int = -1,
                  ind_near: Optional[list[int]] = None,
                  ind_rewire: Optional[list[int]] = None,
                  sampling_ellipses: Optional[list[EllipseParameters]] = None,
                  force_plot: bool = False) -> None:
        """ Plots the rrt plan segment

            Inputs:
                iteration: The iteration number of the planner
                tree: Underlying tree for the plan
                x_start: the start location of the plan
                X_t: target set
                X: state space
                x_rand: The newly sampled point
                x_new: The point that the planner is attempting to add to the tree
                ind_p: The parent index within tree to which x_new it being added
                ind_goal: The index to the lowest cost goal location within the tree
                ind_near: The set of nearest neighbors over which the search is performed
                ind_rewire: The set of nodes that were rewired through x_new
                sampling_ellipses: List of all ellipses used for sampling
                force_plot: True => the plot will be plotted regardless of iteration
        """
        if force_plot or np.mod(iteration, self.plot_iterations) == 0:
            self.fig = plot_rrt(pause_plotting=self.pause_plotting,
                                world=self.world,
                                tree=tree,
                                x_start=x_start,
                                X=X,
                                X_t=X_t,
                                x_rand=x_rand,
                                x_new=x_new,
                                ind_p= ind_p,
                                ind_goal=ind_goal,
                                ind_near=ind_near,
                                ind_rewire=ind_rewire,
                                sampling_ellipses=sampling_ellipses,
                                fig=self.fig)
