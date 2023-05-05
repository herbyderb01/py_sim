"""plotting.py: Plotting utilities
"""
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
from py_sim.tools.sim_types import TwoDArrayType, TwoDimArray

def initialize_position_plot(ax: Axes, color: tuple[float, float, float, float], label: str = "", location: TwoDArrayType = TwoDimArray()) -> Line2D:
    """initialize_position_plot Initializes the plotting of a position and returns a reference
    to the position plot

        Inputs:
            ax: The axis on which to create the plot
            color: The color to plot (rgb-alpha, i.e., color and transparency)
            label: The label to assign to the position
            location: The oiriginal location of the position

        Outputs:
            Reference to the plot (Line2D) for later modification
    """
    (position_plot,) = ax.plot([location.x], [location.y], 'o', label=label, color=color)
    return position_plot

def update_position_plot(line: Line2D, location: TwoDArrayType) -> None:
    """update_position_plot: Updates the position of a previously drawn circle

        Inputs:
            line: Reference to the line to be updated
            location: The (x,y) position of the new location
    """
    line.set_data([location.x], [location.y])