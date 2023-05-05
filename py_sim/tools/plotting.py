"""plotting.py: Plotting utilities
"""
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
import numpy.typing as npt
from py_sim.tools.sim_types import TwoDArrayType, TwoDimArray, UnicyleStateProtocol, UnicycleState

Color = tuple[float, float, float, float] # rgb-alpha color of the plot
blue = (0., 0., 1., 1.)

def initialize_position_plot(ax: Axes, color: Color, label: str = "", location: TwoDArrayType = TwoDimArray()) -> Line2D:
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


class OrientedPositionParams():
    """Contains parameters for plotting an oriented position"""
    def __init__(self, rad: float = 0.2, color: Color = blue) -> None:
        """Initializes the struct of variables used for plotting

            Inputs:
                rad: The radius of the vehicle being plotted
                color: rgb-alpha value for the color of the object
        """
        self.rad = rad
        self.color = color

def create_oriented_polygon(pose: UnicyleStateProtocol, rad: float) -> tuple[list[float], list[float]]:
    """create_oriented_polygon: Returns the oriented polygon to represent a given pose

        Inputs:
            pose: The position and orientation of the vehicle
            rad: The radius of the vehicle

        Returns:
            x: The x-coordinate vector
            y: The y-coordinate vector
    """

    # Calculate the active rotation matrix for the orientation
    c_psi = np.cos(pose.psi)
    s_psi = np.sin(pose.psi)
    R = np.array([[c_psi, -s_psi], [s_psi, c_psi]])

    # Calculate points along the triangle
    q = np.array([[pose.x], [pose.y]])
    p1 = R @ np.array([[-rad], [-rad]]) + q # Right corner
    p2 = q                                  # Tip of triangle
    p3 = R @ np.array([[-rad], [rad]]) + q  # Left corner
    p4a = R @ np.array([[-rad], [0.]]) + q
    p4 = (q+p4a)/2.

    # Extract x and y components
    x = [p1.item(0), p2.item(0), p3.item(0), p4.item(0)]
    y = [p1.item(1), p2.item(1), p3.item(1), p4.item(1)]
    return (x,y)

def init_oriented_position_plot(ax: Axes, params: OrientedPositionParams, pose: UnicyleStateProtocol = UnicycleState())->Polygon:
    """init_oriented_position_plot Creates a triangle-like shape to show the orientaiton of the vehicle

        Inputs:
           ax: The axis on which to create the plot
           params: The parameters defining the plot
           pose: The position and orientation of the vehicle

        Outputs:
            Polygon reference for later updating
    """
    # Calculate the polygon to be plotted
    (x,y) = create_oriented_polygon(pose=pose, rad=params.rad)

    # Create the plot and return a reference
    (pose_plot,) = ax.fill(x, y, color=params.color)
    return pose_plot

def update_oriented_position_plot(plot: Polygon, params: OrientedPositionParams, pose: UnicyleStateProtocol) -> None:
    """update_oriented_position_plot: Updates the plot for the orientated position

        Inputs:
           ax: The axis on which to create the plot
           params: The parameters defining the plot
           pose: The position and orientation of the vehicle
    """
    # Calculate the polygon to be plotted
    (x,y) = create_oriented_polygon(pose=pose, rad=params.rad)
    xy = np.array([x,y]).transpose()

    # Update the plot
    plot.set_xy(xy=xy)
