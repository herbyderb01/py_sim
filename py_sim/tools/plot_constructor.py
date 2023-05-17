"""plot_constructor.py: Constructs plot manifiests
"""
from typing import Optional

import matplotlib.pyplot as plt
from py_sim.tools.plotting import (
    Color,
    PlotManifest,
    PosePlot,
    PositionPlot,
    StateTrajPlot,
    UnicycleTimeSeriesPlot,
    VectorFieldPlot,
)
from py_sim.tools.sim_types import UnicycleStateType, VectorField


def create_plot_manifest(initial_state: UnicycleStateType, # pylint: disable=too-many-arguments
                         y_limits: tuple[float, float],
                         x_limits: tuple[float, float],
                         position_dot: bool = True,
                         position_triangle: bool = True,
                         state_trajectory: bool = True,
                         unicycle_time_series: bool = True,
                         color: Color = (0.2, 0.36, 0.78, 1.0),
                         vectorfield: Optional[VectorField] = None,
                         vector_res: float = 0.25
                         ) -> PlotManifest[UnicycleStateType]:
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
    """
    # Create the manifest to be returned
    plots = PlotManifest[UnicycleStateType]()

    # Create the desired data plots
    if unicycle_time_series:
        state_plot = UnicycleTimeSeriesPlot[UnicycleStateType](color=color)
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

    # Create the desired state plots
    if position_dot:
        plots.state_plots.append(PositionPlot(ax=ax, label="Vehicle", color=color) )
    if position_triangle:
        plots.state_plots.append(PosePlot(ax=ax, rad=0.2))

    # Create the state trajectory plot
    if state_trajectory:
        plots.data_plots.append(StateTrajPlot(ax=ax, label="Vehicle Trajectory", \
                                color=color, location=initial_state))

    # Create a vectorfield plot
    if vectorfield is not None:
        plots.data_plots.append(
            VectorFieldPlot(ax=ax,
                            color=color,
                            y_limits=y_limits,
                            x_limits=x_limits,
                            resolution=vector_res,
                            vector_field=vectorfield))

    return plots
