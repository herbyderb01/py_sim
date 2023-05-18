"""plotting.py: Plotting utilities
"""
from typing import Any, Generic, Optional, Protocol, TypeVar

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from py_sim.tools.sim_types import (
    Data,
    StateType,
    TwoDArrayType,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
    UnicycleStateType,
    UnicyleStateProtocol,
    VectorField,
    RangeBearingMeasurements
)
from py_sim.worlds.polygon_world import PolygonWorld

############################# Plotting Types #######################################
Color = tuple[float, float, float, float] # rgb-alpha color of the plot
blue = (0., 0., 1., 1.)
red = (1., 0., 0., 1.)
green = (0., 1., 0., 1.)

class StatePlot(Protocol[StateType]): # type: ignore
    """Class that defines the plotting framework for a plot requiring state only"""
    def plot(self, state: StateType) -> None:
        """Updates the plot for the given state type"""

class DataPlot(Protocol[StateType]):
    """Class that defines plotting framework for using the full Data"""
    def plot(self, data: Data[StateType]) -> None:
        """Updates the plot given the latest data"""

class PlotManifest(Generic[StateType]):
    """Defines data necessary for plotting"""
    figs: list[Figure] = []
    axes: dict[Axes, Figure] = {}
    state_plots: list[StatePlot[StateType]] = []
    data_plots: list[DataPlot[StateType]] = []


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
LocationStateType = TypeVar("LocationStateType", bound=TwoDArrayType)
class StateTrajPlot(Generic[LocationStateType]):
    """Plots the state trajectory one pose at a time"""
    def __init__(self, ax: Axes, color: Color, location: TwoDArrayType, label: str = "") -> None:
        """ Creates the State Trajectory plot on the given axes

            Inputs:
                ax: The axis on which to create the plot
                color: The color to plot (rgb-alpha, i.e., color and transparency)
                location: The original position of the vehicle
                label: The label to assign to the trajectory plot
        """
        self.handle = initialize_2d_line_plot(ax=ax, color=color, style="-.", x=location.x, y=location.y, label=label)

    def plot(self, data: Data[LocationStateType]) -> None:
        """Update the state trajectory plot"""
        update_2d_line_plot(line=self.handle, x_vec=data.get_state_vec(data.current.state.IND_X),
                            y_vec=data.get_state_vec(data.current.state.IND_Y))

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


###################### Time Series Plot #######################
class UnicycleTimeSeriesPlot(Generic[UnicycleStateType]):
    """Plots the unicycle state vs time with each state in its own subplot"""
    def __init__(self,
                 color: Color,
                 style: str = "-",
                 fig: Optional[Figure] = None,
                 axs: Optional[ list[Axes] ] = None,
                 label: str = "") -> None:
        """Plot a unicycle time series plot

            Inputs:
                color: The color to plot (rgb-alpha, i.e., color and transparency)
                style: The line style of the plot
                fig: The figure on which to plot - If none or axes is none then a new figure is created
                axs: The list of axes on which to plot - If none or fig is non then a new figure is created
                label: The label for the line
        """

        # Create a new figure
        if fig is None or axs is None:
            self.fig, self.axs = plt.subplots(5,1)
        else:
            self.fig = fig
            self.axs = axs

        # Create a new time series for each unicycle state
        self.handle_x = initialize_2d_line_plot(ax=self.axs[0],x=0., y=0., color=color, style=style, label=label)
        self.handle_y = initialize_2d_line_plot(ax=self.axs[1],x=0., y=0., color=color, style=style, label=label)
        self.handle_psi = initialize_2d_line_plot(ax=self.axs[2],x=0., y=0., color=color, style=style, label=label)
        self.handle_v = initialize_2d_line_plot(ax=self.axs[3],x=0., y=0., color=color, style=style, label=label)
        self.handle_w = initialize_2d_line_plot(ax=self.axs[4],x=0., y=0., color=color, style=style, label=label)

        # Label the axes and plots
        self.axs[0].set_ylabel("X position")
        self.axs[1].set_ylabel("Y position")
        self.axs[2].set_ylabel("Orientation")
        self.axs[3].set_ylabel("$u_v$")
        self.axs[4].set_ylabel("$u_\\omega$")
        self.axs[4].set_xlabel("Time (sec)")

    def plot(self, data: Data[UnicycleStateType]) -> None:
        """ Plots the line trajectory
        """
        update_2d_line_plot(line=self.handle_x, x_vec=data.get_time_vec(), y_vec=data.get_state_vec(data.current.state.IND_X))
        update_2d_line_plot(line=self.handle_y, x_vec=data.get_time_vec(), y_vec=data.get_state_vec(data.current.state.IND_Y))
        update_2d_line_plot(line=self.handle_psi, x_vec=data.get_time_vec(), y_vec=data.get_state_vec(data.current.state.IND_PSI))
        update_2d_line_plot(line=self.handle_v, x_vec=data.get_time_vec(), y_vec=data.get_control_vec(UnicycleControl.IND_V))
        update_2d_line_plot(line=self.handle_w, x_vec=data.get_time_vec(), y_vec=data.get_control_vec(UnicycleControl.IND_W))

        # Resize the axis
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view(True, True, True)

class DataTimeSeries(Generic[StateType]):
    """Plots the time series of a given state withing the data object"""
    def __init__(self, ax: Axes, color: Color, state_ind: int, label: str = "") -> None:
        """Creates the State Trajectory plot on the given axes

            Inputs:
                ax: The axis on which to create the plot
                color: The color to plot (rgb-alpha, i.e., color and transparency)
                state_ind: The index of the state being plotted
                label: The label to assign to the trajectory plot
        """
        # Note that the initial (x,y) location will not show up due to the syle
        # and that it will be overwritten each time the plot is updated
        self.handle = initialize_2d_line_plot(ax=ax, color=color, style="-.", x=0., y=0., label=label)
        self.state_ind = state_ind

    def plot(self, data: Data[StateType]) -> None:
        """Update the trajectory plot"""
        update_2d_line_plot(line=self.handle, x_vec=data.get_time_vec(), y_vec=data.get_state_vec(self.state_ind))

def initialize_2d_line_plot(ax: Axes, x: float, y: float, color: Color, style: str = "-", label: str="") -> Line2D:
    """Initializes a two-dimensional line plot

        Inputs:
            ax: The axis on which to create the plot
            x: initial x-position
            y: initial y-position
            color: The color to plot (rgb-alpha, i.e., color and transparency)
            style: The line style of the plot
            label: The label to assign to the trajectory plot

        Outputs:
            Reference to the plot (Line2D) for later modification
    """
    (traj_plot,) = ax.plot([x], [y], style, label=label, color=color)
    return traj_plot

def update_2d_line_plot(line: Line2D, x_vec: npt.NDArray[Any], y_vec: npt.NDArray[Any]) -> None:
    """update_traj_plot: Updates the trajectory with the latest location

        Inputs:
            line: Reference to the line to be updated
            x_vec: The data for the x coordinate
            y_vec: The data for the y coordinate
    """
    line.set_data(x_vec, y_vec)

###################### Vector Field Plot #######################
VectorFieldType = TypeVar("VectorFieldType", bound=VectorField)
class VectorFieldPlot(Generic[VectorFieldType]):
    """Plots vector fields given the current state"""
    def __init__(self,
                 ax: Axes,
                 color: Color,
                 y_limits: tuple[float, float],
                 x_limits: tuple[float, float],
                 resolution: float,
                 vector_field: VectorFieldType) -> None:
        """Creates a vector field plotter

            Inputs:
                ax: The axis on which to create the plot
                color: The color to plot (rgb-alpha, i.e., color and transparency)
                y_limits: min and max y-values to plot
                x_limits: min and max x-values to plot
                resolution: the stepsize for plotting in x and y direction
                vector_field: The vectorfield class to be plotted
        """
        # Store the vectorfield
        self.vector_field: VectorFieldType = vector_field

        # Create the meshgrid of points to be plotted
        x_vec = np.arange(x_limits[0], x_limits[1], resolution)
        y_vec = np.arange(y_limits[0], y_limits[1], resolution)
        self.x_vals, self.y_vals = np.meshgrid(x_vec, y_vec)

        # Create the initial plot
        self.handle = ax.quiver(self.x_vals, self.y_vals, self.x_vals, self.y_vals, color=color, angles='xy', scale_units='xy')

    def plot(self, data: Data[UnicycleStateType]) -> None:
        """Update the pose plot"""

        # Create the grid of velocities (u = x velocity, v = y velocity)
        u_vals = np.zeros(self.x_vals.shape)
        v_vals = np.zeros(self.x_vals.shape)

        # Loop through and calculate the velocity at each point in the meshgrid
        unicycle_state = UnicycleState(x=data.current.state.x, y=data.current.state.y, psi=data.current.state.psi)
        for row in range(self.x_vals.shape[0]):
            for col in range(self.x_vals.shape[1]):
                # Form the state
                unicycle_state.x = self.x_vals[row,col]
                unicycle_state.y = self.y_vals[row,col]

                # Create the vector
                vec = self.vector_field.calculate_vector(state=unicycle_state, time=data.current.time)

                # Store the vector
                u_vals[row,col] = vec.x
                v_vals[row,col] = vec.y

        # Update the plot
        self.handle.set_UVC(U=u_vals, V=v_vals)

###################### World Plotters #########################
def plot_polygon_world(ax: Axes, world: PolygonWorld,  color: Color = red)->list[Polygon]:
    """Plots each of the polygons in the world

        Inputs:
            ax: axis on which the world should be plotted
            world: the instance of the world to be plotted
            color: the color of the obstacles

        Returns:
            A list of all of the polygon plot objects
    """
    # Loop through and plot each polygon individually
    handles: list[Polygon] = []
    for polygon in world.polygons:
        (handle, ) = ax.fill(polygon.points[0,:],polygon.points[1,:], color=color)
        handles.append(handle)

    return handles


##################### Range Bearing Plotter ###################
class RangeBearingPlot(Generic[LocationStateType]):
    """Plots the state trajectory one pose at a time"""
    def __init__(self, ax: Axes, color: Color = green, label: str = "") -> None:
        """ Creates the State Trajectory plot on the given axes

            Inputs:
                ax: The axis on which to create the plot
                color: The color to plot (rgb-alpha, i.e., color and transparency)
                location: The original position of the vehicle
                label: The label to assign to the trajectory plot
        """
        self.handle = initialize_2d_line_plot(ax=ax, color=color, style="o", x=0., y=0., label=label)

    def plot(self, data: Data[LocationStateType]) -> None:
        """Update the state trajectory plot"""
        # Extract the bearing locations for the data that has a measurement
        x_vec: list[float] = []
        y_vec: list[float] = []
        for (location, dist) in zip(data.range_bearing_latest.location, data.range_bearing_latest.range):
            if dist < np.inf:
                x_vec.append(location.x)
                y_vec.append(location.y)

        update_2d_line_plot(line=self.handle, x_vec=x_vec,
                            y_vec=y_vec)