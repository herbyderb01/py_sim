"""plotting.py: Plotting utilities
"""
from matplotlib.axes._axes import Axes
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
import numpy as np
from typing import Protocol
from py_sim.tools.sim_types import TwoDArrayType, TwoDimArray, UnicyleStateProtocol, UnicycleState
from py_sim.tools.sim_types import StateType

Color = tuple[float, float, float, float] # rgb-alpha color of the plot
blue = (0., 0., 1., 1.)

class StatePlot(Protocol[StateType]): # type: ignore
    def plot(self, state: StateType) -> None:
        """Updates the plot for the given state type"""

############################# 2D Position Plot Object ##############################
class PositionPlot():
    """Plots the position as a circle"""
    def __init__(self, ax: Axes, color: Color, label: str = "", location: TwoDArrayType = TwoDimArray()) -> None:
        """Initailizes a position plot given the desired attributes

            Inputs:
                ax: The axis on which to create the plot
                color: The color to plot (rgb-alpha, i.e., color and transparency)
                label: The label to assign to the position
                location: The oiriginal location of the position
        """
        self.position_plot = initialize_position_plot(ax=ax, color=color, location=location, label=label)

    def plot(self, state: TwoDArrayType) -> None:
        """ Updates the plot given the new 2D position
        """
        update_position_plot(line=self.position_plot, location=state)

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


############################# Pose Plot Object ##############################
class PosePlot():
    """Plotes the position and orientation as a triangular like object"""
    def __init__(self, ax: Axes, rad: float = 0.2, color: Color = blue) -> None:
        """Initailizes a plot of the pose"""
        self.params = OrientedPositionParams(rad=rad, color=color)
        self.poly_plot = init_oriented_position_plot(ax=ax, params=self.params)

    def plot(self, state: UnicyleStateProtocol) -> None:
        """Update the pose plot"""
        update_oriented_position_plot(plot=self.poly_plot, params=self.params, pose=state)

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


############################# State Trajectory Plot Object ##############################
class StateTrajPlot():
    """Plots the state trajectory one pose at a time"""
    def __init__(self, ax: Axes, color: Color, location: TwoDArrayType, label: str = "") -> None:
        """ Creates the State Trajectory plot on the given axes

            Inputs:
                ax: The axis on which to create the plot
                color: The color to plot (rgb-alpha, i.e., color and transparency)
                location: The original position of the vehicle
                label: The label to assign to the trajectory plot
        """
        self.handle = initialize_traj_plot(ax=ax, color=color, label=label, location=location)

    def plot(self, state: TwoDArrayType) -> None:
        """Update the state trajectory plot"""
        update_traj_plot(line=self.handle, location=state)

def initialize_traj_plot(ax: Axes, color: Color, label: str = "", location: TwoDArrayType = TwoDimArray()) -> Line2D:
    """initialize_traj_plot Initializes the plotting of a trajectory and returns a reference
    to the trajectory plot

        Inputs:
            ax: The axis on which to create the plot
            color: The color to plot (rgb-alpha, i.e., color and transparency)
            label: The label to assign to the trajectory plot
            location: The original position of the vehicle

        Outputs:
            Reference to the plot (Line2D) for later modification
    """
    (traj_plot,) = ax.plot([location.x], [location.y], '-.', label=label, color=color)
    return traj_plot

def update_traj_plot(line: Line2D, location: TwoDArrayType) -> None:
    """update_traj_plot: Updates the trajectory with the latest location

        Inputs:
            line: Reference to the line to be updated
            location: The (x,y) position of the new location
    """
    # Get the previously plottied data
    x = line.get_xdata()
    y = line.get_ydata()

    # Add in the new data
    x = np.append(x, location.x)
    y = np.append(y, location.y)
    line.set_data(x, y)
