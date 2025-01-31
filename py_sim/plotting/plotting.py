"""plotting.py: Plotting utilities

Attributes:
    Color(tuple[float, float, float, float]): Defines a type for the color tuple
    blue(tuple[float, float, float, float]): Quick color definition
    red(tuple[float, float, float, float]): Quick color definition
    green(tuple[float, float, float, float]): Quick color definition
    black(tuple[float, float, float, float]): Quick color definition
    white(tuple[float, float, float, float]): Quick color definition

    LocationStateType(TypeVar): Defines a state with a location bound by TwoDArrayType
    VectorFieldType(TypeVar): Defines a type correpsonding to a vector field to be plotted
"""
from typing import Any, Generic, Optional, Protocol, TypeVar, cast

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.image import AxesImage
from matplotlib.lines import Line2D
from matplotlib.patches import Polygon
from py_sim.dynamics.unicycle import solution_trajectory as uni_soln_traj
from py_sim.sensors.occupancy_grid import (
    BinaryOccupancyGrid,
    ind2sub,
    occupancy_positions,
)
from py_sim.tools.projections import LineCarrot
from py_sim.tools.sim_types import (
    Data,
    LocationStateType,
    StateType,
    TwoDArrayType,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
    UnicycleStateProtocol,
    UnicycleStateType,
    VectorField,
)
from py_sim.tools.simple_priority_queue import SimplePriorityQueue
from py_sim.worlds.polygon_world import PolygonWorld

############################# Plotting Types #######################################
Color = tuple[float, float, float, float] # rgb-alpha color of the plot
blue = (0., 0., 1., 1.)
red = (1., 0., 0., 0.9)
green = (0., 1., 0., 1.)
black = (0., 0., 0., 0.5)
white = (1., 1., 1., 0.5)

class StatePlot(Protocol[StateType]): # type: ignore
    """Class that defines the plotting framework for a plot requiring state only"""
    def plot(self, state: StateType) -> None:
        """Updates the plot for the given state type
        Args:
            state: The state to be used for the update
        """

class DataPlot(Protocol[StateType]):
    """Class that defines plotting framework for using the full Data"""
    def plot(self, data: Data[StateType]) -> None:
        """Updates the plot given the latest data

        Args:
            data: The data to be used for the plot update
        """

class PlotManifest(Generic[StateType]):
    """Defines data necessary for plotting

    Attributes:
        figs(list[Figure]): Figures created for plotting
        vehicle_axes(Axes): Stores the axes for the vehicle plot
        state_plots(list[StatePlot[StateType]]): List of all the state plots
            (plots that only depend on the state)
        data_plots(list[DataPlot[StateType]]): List of all the data plots
            (plots that depend on many data elements)
    """
    figs: list[Figure] = []
    vehicle_axes: Axes
    state_plots: list[StatePlot[StateType]] = []
    data_plots: list[DataPlot[StateType]] = []


############################# 2D Position Plot Object ##############################
class PositionPlot():
    """Plots the position as a circle

    Attributes:
        position_plot(Line2D): Reference to the position plot
    """
    def __init__(self, ax: Axes, color: Color, label: str = "", location: TwoDArrayType = TwoDimArray()) -> None:
        """Initailizes a position plot given the desired attributes

        Args:
            ax: The axis on which to create the plot
            color: The color to plot (rgb-alpha, i.e., color and transparency)
            label: The label to assign to the position
            location: The oiriginal location of the position
        """
        self.position_plot = initialize_position_plot(ax=ax, color=color, location=location, label=label)

    def plot(self, state: TwoDArrayType) -> None:
        """ Updates the plot given the new 2D position

        Args:
            state: State to be plotted
        """
        update_position_plot(line=self.position_plot, location=state)

def initialize_position_plot(ax: Axes, color: Color, label: str = "", location: TwoDArrayType = TwoDimArray()) -> Line2D:
    """initialize_position_plot Initializes the plotting of a position and returns a reference
    to the position plot

        Args:
            ax: The axis on which to create the plot
            color: The color to plot (rgb-alpha, i.e., color and transparency)
            label: The label to assign to the position
            location: The oiriginal location of the position

        Returns:
            Line2D: Reference to the plot (Line2D) for later modification
    """
    (position_plot,) = ax.plot([location.x], [location.y], 'o', label=label, color=color)
    return position_plot

def update_position_plot(line: Line2D, location: TwoDArrayType) -> None:
    """update_position_plot: Updates the position of a previously drawn circle

    Args:
        line: Reference to the line to be updated
        location: The (x,y) position of the new location
    """
    line.set_data([location.x], [location.y])


############################# Pose Plot Object ##############################
class PosePlot():
    """Plotes the position and orientation as a triangular like object

    Attributes:
        params(OrientedPositionParams): The parameters needed for plotting the pose
        poly_plot(Polygon): Reference to the polygon
    """
    def __init__(self, ax: Axes, rad: float = 0.2, color: Color = blue) -> None:
        """Initailizes a plot of the pose

        Args:
            ax: Axes on which to plot the position
            rad: Radius of the position being plotted
            color: The color to plot the vehicle
        """
        self.params = OrientedPositionParams(rad=rad, color=color)
        self.poly_plot = init_oriented_position_plot(ax=ax, params=self.params)

    def plot(self, state: UnicycleStateProtocol) -> None:
        """Update the pose plot"""
        update_oriented_position_plot(plot=self.poly_plot, params=self.params, pose=state)

class OrientedPositionParams():
    """Contains parameters for plotting an oriented position

    Attributes:
        rad(float): Radius of the position being plotted
        color(Color): Color of the plotted position
    """
    def __init__(self, rad: float = 0.2, color: Color = blue) -> None:
        """Initializes the struct of variables used for plotting

        Args:
            rad: The radius of the vehicle being plotted
            color: rgb-alpha value for the color of the object
        """
        self.rad = rad
        self.color = color

def create_oriented_polygon(pose: UnicycleStateProtocol, rad: float) -> tuple[list[float], list[float]]:
    """create_oriented_polygon: Returns the oriented polygon to represent a given pose

    Args:
        pose: The position and orientation of the vehicle
        rad: The radius of the vehicle

    Returns:
        tuple[list[float], list[float]]:
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

def init_oriented_position_plot(ax: Axes, params: OrientedPositionParams, pose: UnicycleStateProtocol = UnicycleState())->Polygon:
    """init_oriented_position_plot Creates a triangle-like shape to show the orientaiton of the vehicle

        Args:
           ax: The axis on which to create the plot
           params: The parameters defining the plot
           pose: The position and orientation of the vehicle

        Returns:
            Polygon: Polygon reference for later updating
    """
    # Calculate the polygon to be plotted
    (x,y) = create_oriented_polygon(pose=pose, rad=params.rad)

    # Create the plot and return a reference
    (pose_plot,) = ax.fill(x, y, color=params.color)
    return pose_plot

def update_oriented_position_plot(plot: Polygon, params: OrientedPositionParams, pose: UnicycleStateProtocol) -> None:
    """update_oriented_position_plot: Updates the plot for the orientated position

    Args:
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
class StateTrajPlot(Generic[LocationStateType]):
    """Plots the state trajectory one pose at a time

    Attributes:
        handle(Line2D): Reference to the line being plotted
    """
    def __init__(self, ax: Axes, color: Color, location: TwoDArrayType, label: str = "") -> None:
        """ Creates the State Trajectory plot on the given axes

            Args:
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

    Args:
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
class UnicycleTimeSeriesPlot():
    """Plots the unicycle state vs time with each state in its own subplot

    fig(Figure): Figure on which the trajectories are plot
    axs(np.ndarray[Axes]): The subplot axes on which everything is plot
    handle_x(Line2D): reference to the line on which the x state is plot
    handle_y(Line2D): reference to the line on which the y state is plot
    handle_psi(Line2D): reference to the line on which the psi state is plot
    handle_v(Line2D): reference to the line on which the v state is plot
    handle_w(Line2D): reference to the line on which the w state is plot
    """
    def __init__(self,
                 color: Color,
                 style: str = "-",
                 fig: Optional[Figure] = None,
                 axs: Optional[ list[Axes] ] = None,
                 label: str = "") -> None:
        """Plot a unicycle time series plot

        Args:
            color: The color to plot (rgb-alpha, i.e., color and transparency)
            style: The line style of the plot
            fig: The figure on which to plot - If none or axes is none then a new figure is created
            axs: The list of axes on which to plot - If none or fig is non then a new figure is created
            label: The label for the line
        """

        # Create a new figure
        self.axs: list[Axes]
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
        self.axs[0].set_ylabel("X")
        self.axs[1].set_ylabel("Y")
        self.axs[2].set_ylabel("Orien")
        self.axs[3].set_ylabel("$u_v$")
        self.axs[4].set_ylabel("$u_\\omega$")
        self.axs[4].set_xlabel("Time (sec)")

    def plot(self, data: Data[UnicycleState]) -> None:
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

class SingleIntegratorTimeSeriesPlot():
    """Plots the unicycle state vs time with each state in its own subplot

    fig(Figure): Figure on which the trajectories are plot
    axs(npt.NDArray(Axes)): The subplot axes on which everything is plot
    handle_x(Line2D): reference to the line on which the x state is plot
    handle_y(Line2D): reference to the line on which the y state is plot
    handle_psi(Line2D): reference to the line on which the psi state is plot
    handle_v(Line2D): reference to the line on which the v state is plot
    handle_w(Line2D): reference to the line on which the w state is plot
    """
    def __init__(self,
                 color: Color,
                 style: str = "-",
                 fig: Optional[Figure] = None,
                 axs: Optional[ list[Axes] ] = None,
                 label: str = "") -> None:
        """Plot a unicycle time series plot

        Args:
            color: The color to plot (rgb-alpha, i.e., color and transparency)
            style: The line style of the plot
            fig: The figure on which to plot - If none or axes is none then a new figure is created
            axs: The list of axes on which to plot - If none or fig is non then a new figure is created
            label: The label for the line
        """

        # Create a new figure
        self.axs: list[Axes]
        if fig is None or axs is None:
            self.fig, self.axs = plt.subplots(4,1)
        else:
            self.fig = fig
            self.axs = axs

        # Create a new time series for each unicycle state
        self.handle_x = initialize_2d_line_plot(ax=self.axs[0],x=0., y=0., color=color, style=style, label=label)
        self.handle_y = initialize_2d_line_plot(ax=self.axs[1],x=0., y=0., color=color, style=style, label=label)
        self.handle_xdot = initialize_2d_line_plot(ax=self.axs[2],x=0., y=0., color=color, style=style, label=label)
        self.handle_ydot = initialize_2d_line_plot(ax=self.axs[3],x=0., y=0., color=color, style=style, label=label)

        # Label the axes and plots
        self.axs[0].set_ylabel("X")
        self.axs[1].set_ylabel("Y")
        self.axs[2].set_ylabel("$\\dot{x}$")
        self.axs[3].set_ylabel("$\\dot{y}$")
        self.axs[3].set_xlabel("Time (sec)")

    def plot(self, data: Data[LocationStateType]) -> None:
        """ Plots the line trajectory
        """
        update_2d_line_plot(line=self.handle_x, x_vec=data.get_time_vec(), y_vec=data.get_state_vec(data.current.state.IND_X))
        update_2d_line_plot(line=self.handle_y, x_vec=data.get_time_vec(), y_vec=data.get_state_vec(data.current.state.IND_Y))
        update_2d_line_plot(line=self.handle_xdot, x_vec=data.get_time_vec(), y_vec=data.get_control_vec(TwoDimArray.IND_X))
        update_2d_line_plot(line=self.handle_ydot, x_vec=data.get_time_vec(), y_vec=data.get_control_vec(TwoDimArray.IND_Y))

        # Resize the axis
        for ax in self.axs:
            ax.relim()
            ax.autoscale_view(True, True, True)

class DataTimeSeries(Generic[StateType]):
    """Plots the time series of a given state withing the data object

    Attributes:
        handle(Line2D): Reference to the line being plot
        state_ind(int): Index to the state within the state vector that is being plot
    """
    def __init__(self, ax: Axes, color: Color, state_ind: int, label: str = "") -> None:
        """Creates the State Trajectory plot on the given axes

        Args:
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

    Args:
        ax: The axis on which to create the plot
        x: initial x-position
        y: initial y-position
        color: The color to plot (rgb-alpha, i.e., color and transparency)
        style: The line style of the plot
        label: The label to assign to the trajectory plot

    Returns:
        Line2D: Reference to the plot (Line2D) for later modification
    """
    (traj_plot,) = ax.plot([x], [y], style, label=label, color=color)
    return traj_plot

def update_2d_line_plot(line: Line2D, x_vec: npt.NDArray[Any], y_vec: npt.NDArray[Any]) -> None:
    """update_traj_plot: Updates the trajectory with the latest location

    Args:
        line: Reference to the line to be updated
        x_vec: The data for the x coordinate
        y_vec: The data for the y coordinate
    """
    line.set_data(x_vec, y_vec)


###################### Vector Field Plot #######################
VectorFieldType = TypeVar("VectorFieldType", bound=VectorField)
class VectorFieldPlot(Generic[VectorFieldType]):
    """Plots vector fields given the current state

    Attributes:
        vector_field(VectorFieldType): The vectorfield class to be plotted
        x_vals(NDArray[Any]): The x positions of all the elements being plot
        y_vals(NDArray[Any]): The y positions of all the elements being plot
        handle(Quiver): The reference to the quivers being plotted
    """
    def __init__(self,
                 ax: Axes,
                 color: Color,
                 y_limits: tuple[float, float],
                 x_limits: tuple[float, float],
                 resolution: float,
                 vector_field: VectorFieldType) -> None:
        """Creates a vector field plotter

            Args:
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
        # self.handle = ax.quiver(self.x_vals, self.y_vals, self.x_vals, self.y_vals, color=color, angles='xy', scale_units='xy', scale = 1.)

    def plot(self, data: Data[LocationStateType]) -> None:
        """Update the pose plot"""

        # Create the grid of velocities (u = x velocity, v = y velocity)
        u_vals = np.zeros(self.x_vals.shape)
        v_vals = np.zeros(self.x_vals.shape)

        # Loop through and calculate the velocity at each point in the meshgrid
        position = TwoDimArray(x=data.current.state.x, y=data.current.state.y)
        for row in range(self.x_vals.shape[0]):
            for col in range(self.x_vals.shape[1]):
                # Form the state
                position.x = self.x_vals[row,col]
                position.y = self.y_vals[row,col]

                # Create the vector
                vec = self.vector_field.calculate_vector(state=position, time=data.current.time)

                # Store the vector
                u_vals[row,col] = vec.x
                v_vals[row,col] = vec.y

        # Update the plot
        self.handle.set_UVC(U=u_vals, V=v_vals)

###################### World Plotters #########################
def plot_polygon_world(ax: Axes, world: PolygonWorld,  color: Color = red)->list[Polygon]:
    """Plots each of the polygons in the world

    Args:
        ax: axis on which the world should be plotted
        world: the instance of the world to be plotted
        color: the color of the obstacles

    Returns:
        list[Polygon]: A list of all of the polygon plot objects
    """
    # Loop through and plot each polygon individually
    handles: list[Polygon] = []
    for polygon in world.polygons:
        (handle, ) = ax.fill(polygon.points[0,:],polygon.points[1,:], color=color)
        handles.append(handle)

    return handles


##################### Range Bearing Plotter ###################
class RangeBearingPlot(Generic[LocationStateType]):
    """Plots the positions of detections be the range bearing sensors

    Attributes:
        handle(Line2D): reference to the line being plotted
    """
    def __init__(self, ax: Axes, color: Color = green, label: str = "") -> None:
        """ Creates the State Trajectory plot on the given axes

            Args:
                ax: The axis on which to create the plot
                color: The color to plot (rgb-alpha, i.e., color and transparency)
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

        update_2d_line_plot(line=self.handle,
                            x_vec=cast(npt.NDArray[Any], x_vec),
                            y_vec=cast(npt.NDArray[Any], y_vec) )

class RangeBearingLines(Generic[LocationStateType]):
    """Plots the positions of detections be the range bearing sensors

    Attributes:
        ax(Axes): The axis on which to create the plot
        color(Color): The color to plot (rgb-alpha, i.e., color and transparency)
        label(str): The label to assign to the trajectory plot
        handles(list[Line2D]): The references to each of the sensor lines

    """
    def __init__(self, ax: Axes, color: Color = green, label: str = "") -> None:
        """ Creates the State Trajectory plot on the given axes

            Args:
                ax: The axis on which to create the plot
                color: The color to plot (rgb-alpha, i.e., color and transparency)
                label: The label to assign to the trajectory plot
        """
        self.ax = ax
        self.color = color
        self.label = label
        self.handles: list[Line2D] = []

    def initialize_line_plots(self, data: Data[LocationStateType]) -> None:
        """Initailizes the plots of the lines given the number of lines"""
        self.handles = []
        for _ in data.range_bearing_latest.location:
            self.handles.append(
                initialize_2d_line_plot(ax=self.ax, color=self.color, style="-", x=0., y=0., label=self.label)
            )

    def plot(self, data: Data[LocationStateType]) -> None:
        """Update the state trajectory plot"""
        # Initialize the line plots if not done yet
        if len(self.handles) < 1:
            self.initialize_line_plots(data=data)

        # Plot the line for each sensor measurement
        pose = data.current.state
        for handle, location in zip(self.handles, data.range_bearing_latest.location):
            x_vec = np.array([pose.x, location.x])
            y_vec = np.array([pose.y, location.y])
            update_2d_line_plot(line=handle,
                                x_vec=x_vec,
                                y_vec=y_vec)

##################### Occupancy Grid Plotter ###################
def plot_occupancy_grid_circles(ax: Axes,
                                grid: BinaryOccupancyGrid,
                                color_occupied: Color = black,
                                color_free: Optional[Color] = None) -> tuple[Line2D, Optional[Line2D]]:
    """ Plots the occupancy grid as circles

    Args:
        ax: axis on which the grid should be plotted
        grid: grid to be plotted
        color_occupied: color of the occupied regions
        color_free: color of the free regions. If not provided then the free regions are not plotted

    Returns:
        tuple[Line2D, Optional[Line2D]]: plot handles for occupied and free plots
    """
    # Calculate the positions for occupied and free regions
    (x_occ, y_occ, x_free, y_free) = occupancy_positions(grid=grid)

    # Plot the free locations (free plotted first so that the obstacles are apparent on top)
    handle_free: Optional[Line2D] = None
    if color_free is not None:
        (handle_free,) = ax.plot(x_free, y_free, 'o', color=color_free)


    # Plot the occupied locations
    (handle_occ,) = ax.plot(x_occ, y_occ, 'o', color=color_occupied)

    return (handle_occ, handle_free)

def plot_occupancy_grid_cells(ax: Axes,
                              grid: BinaryOccupancyGrid,
                              color_occupied: Color = black,
                              color_free: Color = white) -> list[Polygon]:
    """ Plots the occupancy grid as circles

    Args:
        ax: axis on which the grid should be plotted
        grid: grid to be plotted
        color_occupied: color of the occupied regions
        color_free: color of the free regions.

    Returns:
        list[Polygon]: plot handles for occupied and free plots, each one being a cell
    """
    # Initialize the output
    polygons: list[Polygon] = []

    # Loop through and plot each cell of the occupancy grid
    n_cells = grid.n_cols*grid.n_rows
    for ind in range(n_cells):
        # Get the positions of the cell
        row, col = ind2sub(grid.n_cols, ind)
        x, y = grid.get_cell_box(row=row, col=col)

        # Determine the cell color by occupancy
        cell_color: Color
        if grid.grid[row, col] == grid.FREE:
            cell_color = color_free
        else:
            cell_color = color_occupied

        # Create the cell plot
        polygons.extend(ax.fill(x, y, color=cell_color))

    # Return the result
    return polygons

def plot_occupancy_grid_image(ax: Axes, grid: BinaryOccupancyGrid) -> AxesImage:
    """Plots the occupancy grid as an image as well as the grid lines. This is much faster
       then plotting the grid cells individually, especially for large occupancy grids.

    Args:
        ax: axis on which the grid should be plotted
        grid: grid to be plotted

    Returns:
        AxesImage: The handle for the image used to plot the occupancy map
    """
    handle = ax.imshow( grid.grid,
                        origin="upper",
                        extent=(grid.x_lim[0]-grid.res, grid.x_lim[1]-grid.res, grid.y_lim[0]+grid.res, grid.y_lim[1]+grid.res),
                        cmap="Greys")
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(grid.x_lim[0], grid.x_lim[1], grid.res))
    ax.set_yticks(np.arange(grid.y_lim[0], grid.y_lim[1], grid.res))
    plt.tick_params(axis='both', labelsize=0, length = 0)
    return handle

##################### Plan Plotting #########################################
class GridVisitedType(Protocol):
    """Protocol defining exactly what the Plan Visited needs for plotting

    Attributes:
        grid(BinaryOccupancyGrid): Occupancy grid for the planning space dimensions
        visited(NDArray[Any]): Boolean grid for visited locations same size as grid
    """
    grid: BinaryOccupancyGrid # Occupancy grid for the planning space dimensions
    visited: npt.NDArray[Any] # Boolean grid for visited locations same size as grid

class PlanVisitedPlotter(Generic[LocationStateType]):
    """Plots the visited positions of a graph forward search planner

    Attributes:
        handle_visited(Line2D): The reference to the positions that have been visited
        planner(GridVisitedType): Reference to the grid/visited locations
    """
    def __init__(self, ax: Axes,
                       planner: GridVisitedType,
                       color: Color = black) -> None:
        """
            Args:
                ax: axis on which the visited nodes should be plotted
                planner: planner to be plotted
                color: color of the occupied regions
        """
        super().__init__()
        # Calculate the positions for occupied and free regions
        (x_vis, y_vis, _, __) = occupancy_positions(grid=planner.grid, cells=planner.visited)

        # Plot the occupied locations
        (self.handle_visited,) = ax.plot(x_vis, y_vis, 'o', color=color)
        self.planner = planner

    def plot(self, data: Data[LocationStateType]) -> None: # pylint: disable=unused-argument
        """Update the visited positions plot"""
        # Calculate the positions for occupied and free regions
        (x_vis, y_vis, _, __) = occupancy_positions(grid=self.planner.grid, cells=self.planner.visited)

        # Update the plotter positions
        update_2d_line_plot(line=self.handle_visited,
                            x_vec=cast(npt.NDArray[Any], x_vis),
                            y_vec=cast(npt.NDArray[Any], y_vis))

class QueueType(Protocol):
    """Protocol defining exactly what is needed for plotting the queue

    Attributes:
        grid(BinaryOccupancyGrid): Occupancy grid for the planning space dimensions
        queue(SimplePriorityQueue): The queue for active nodes being considered
    """
    grid: BinaryOccupancyGrid
    queue: SimplePriorityQueue

class PlanQueuePlotter(Generic[LocationStateType]):
    """Plots the positions of the elements in the planner queue

    Attributes:
        planner(QueueType): The planner being used for plotting
        handle_queue(Line2D): The reference to the positions being plot
    """
    def __init__(self, ax: Axes,
                       planner: QueueType,
                       color: Color = green) -> None:
        """ Initilize the queue plotter

        Args:
            ax: Axes on which the plotting will occur
            planner: The path planner being used for planning
            color: The color that the queue locations will be plot
        """
        super().__init__()

        # Store the planner
        self.planner = planner

        # Plot the queue locations
        (self.handle_queue,) = ax.plot([0.], [0.], 'o', color=color)

    def plot(self, data: Data[LocationStateType]) -> None: # pylint: disable=unused-argument
        """Update the queue of elements being plotted"""
        # Calculate the locations of the elements in the queue
        x_vec: list[float] = []
        y_vec: list[float] = []
        for (_, ind) in self.planner.queue.q:
            # Get the corresponding row and column
            (row, col) = ind2sub(n_cols=self.planner.grid.n_cols, ind=ind)

            # Get the position
            position = self.planner.grid.indices_to_position(row=row, col=col)
            x_vec.append(position.x)
            y_vec.append(position.y)

        # Plot the resulting positions
        update_2d_line_plot(line=self.handle_queue,
                            x_vec=cast(npt.NDArray[Any], x_vec),
                            y_vec=cast(npt.NDArray[Any], y_vec))

class PlanType(Protocol):
    """Protocol defining exactly what is needed to plot the plan

    Attributes:
        grid(BinaryOccupancyGrid): Occupancy grid for the planning space dimensions
        parent_mapping(dict[int, int]): Stores the mapping from an index to its parent
    """
    grid: BinaryOccupancyGrid
    parent_mapping: dict[int, int]
    def get_plan_cartesian(self, end_index: Optional[int] = None) -> tuple[list[float], list[float]]:
        """Returns the (x,y) cartesian coordinates of each point along the plan. Assumes that
           planning has already been performed. Throws a ValueError if the end index cannot be connected to the starting index

            Args:
                end_index: The index to which to plan. If None, then the plan end index will be used

            Returns:
                tuple[list[float], list[float]]: A list of x and y coordinates for each point along the plan
        """

class PlanPlotter(Generic[LocationStateType]):
    """Plots the resulting plan to the goal location

    Attributes:
        planner(PlanType): reference to the planner used for planning
        ind_end(int): The index of the goal position
        handle_path(Line2D): Reference to the path being plotted

    """
    def __init__(self, ax: Axes,
                       planner: PlanType,
                       ind_start: int,
                       ind_end: int,
                       color: Color = blue) -> None:
        """
        Args:
            planner: reference to the planner used for planning
            ind_start: The starting index
            ind_end: The ending index of the plan
            color: color for the plot
        """
        super().__init__()

        # Store the planning variables
        self.planner = planner
        self.ind_end = ind_end

        # Gets the initial location
        (row, col) = ind2sub(n_cols=self.planner.grid.n_cols, ind=ind_start)
        position = self.planner.grid.indices_to_position(row=row, col=col)

        # Plot the occupied locations
        (self.handle_path,) = ax.plot([position.x], [position.y], 'o', color=color)

    def plot(self, data: Data[LocationStateType]) -> None: # pylint: disable=unused-argument
        """Update the queue of elements being plotted"""
        # Determine if there is a plan to the end point yet
        if self.ind_end not in self.planner.parent_mapping:
            return

        # Calculate the locations of the elements in the plan
        (x_vec, y_vec) = self.planner.get_plan_cartesian(end_index=self.ind_end)

        # Plot the resulting positions
        update_2d_line_plot(line=self.handle_path,
                            x_vec=cast(npt.NDArray[Any], x_vec),
                            y_vec=cast(npt.NDArray[Any], y_vec))

class PlanVisitedGridPlotter:
    """Plots the visited positions of a graph forward search planner as well as the occupancy grid as a grid

    Attributes:
        handle_im(AxesImage): Reference to the image of the visited elements
        planner(GridVisitedType): planner to be plotted
    """
    def __init__(self, ax: Axes,
                       planner: GridVisitedType) -> None:
        """
            Args:
                ax: axis on which the visited nodes should be plotted
                planner: planner to be plotted
                color: color of the occupied regions
        """
        super().__init__()
        self.handle_im = plot_visited(ax=ax, planner=planner)
        self.planner = planner

    def plot(self, data: Data[LocationStateType]) -> None: # pylint: disable=unused-argument
        """Update the visited positions plot"""
        self.handle_im.set_data(self.planner.grid.grid + 0.5*self.planner.visited)

def plot_visited(ax: Axes, planner: GridVisitedType) -> AxesImage:
    """Plots the occupancy grid as an image as well as the grid lines. This is much faster
       then plotting the grid cells individually, especially for large occupancy grids.

    Args:
        ax: axis on which the grid should be plotted
        grid: grid to be plotted

    Returns:
        AxesImage: The handle for the image used to plot the occupancy map
    """
    handle = ax.imshow( planner.grid.grid + 0.5*planner.visited,
                        origin="upper",
                        extent=(planner.grid.x_lim[0]-planner.grid.res, planner.grid.x_lim[1] \
                                -planner.grid.res, planner.grid.y_lim[0]+planner.grid.res, \
                                planner.grid.y_lim[1]+planner.grid.res),
                        cmap="BuGn")
    ax.grid(which='major', axis='both', linestyle='-', color='k', linewidth=1)
    ax.set_xticks(np.arange(planner.grid.x_lim[0], planner.grid.x_lim[1], planner.grid.res))
    ax.set_yticks(np.arange(planner.grid.y_lim[0], planner.grid.y_lim[1], planner.grid.res))
    plt.tick_params(axis='both', labelsize=0, length = 0)
    return handle

class CarrotPositionPlot():
    """Plots the position of a Carrot as a circle

    Attributes:
        position_plot(Line2D): Reference to the position plot
        carrot(LineCarrot): Reference to the carrot point calculator
    """
    def __init__(self, ax: Axes, color: Color, carrot: LineCarrot, label: str = "", location: TwoDArrayType = TwoDimArray()) -> None:
        """Initailizes a position plot given the desired attributes

        Args:
            ax: The axis on which to create the plot
            color: The color to plot (rgb-alpha, i.e., color and transparency)
            carrot: The carrot of the vehicle
            label: The label to assign to the position
            location: The oiriginal location of the position
        """
        self.position_plot = initialize_position_plot(ax=ax, color=color, location=location, label=label)
        self.carrot = carrot

    def plot(self, state: TwoDArrayType) -> None:
        """ Updates the plot given the new 2D position

        Args:
            state: State to be plotted
        """
        # Plot the point
        update_position_plot(line=self.position_plot,
                             location=self.carrot.get_carrot_point(point=state))

class ControlArcPlot(Generic[UnicycleStateType]):
    """Plots a the arc resulting from a constant unicycle control input

    Attributes:
        ax(Axes): The axis on which to create the plot
        handle(Line2D): The reference to the arc plot
    """
    def __init__(self,
                 ax: Axes,
                 color: Color = green,
                 label: str = "") -> None:
        """Initializes the arc plotter

        Args:
            ax: The axis on which to create the plot
            delta_t: The time length of the arc
            color: The color to plot (rgb-alpha, i.e., color and transparency)
            label: The label to assign to the plot
        """
        super().__init__()
        self.ax = ax
        (self.handle,) = ax.plot([0.], [0.], color=color, label=label)

    def plot(self,
             state: UnicycleStateType,
             control: UnicycleControl,
             ds: float,
             tf: float ) -> None:
        """Plots the arc given the current state

        Args:
            state: The current state of the vehicle
            control: The control input defining the arc
            ds: The step size of the arc (meters)
            tf: The time length of the arc (seconds)
        """

        # Generate state trajectory using unicycle_solution
        (x_vec, y_vec) = uni_soln_traj(init=state,
                                       control = control,
                                       ds=ds,
                                       tf=tf)

        # Plot the state trajectory
        self.handle.set_data(x_vec, y_vec)
