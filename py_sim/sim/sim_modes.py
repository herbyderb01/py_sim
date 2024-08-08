""" Defines a number of simulation modes that can be used
"""

import copy
from threading import Event, Lock
from typing import Generic, Optional

import numpy as np
import py_sim.plotting.plotting as sim_plt
from py_sim.path_planning import dwa
from py_sim.plotting.plotting import PlotManifest
from py_sim.sensors.range_bearing import RangeBearingSensor
from py_sim.sim.generic_sim import SimParameters
from py_sim.sim.integration import euler_update
from py_sim.tools.projections import LineCarrot
from py_sim.tools.sim_types import (
    ArcParams,
    Control,
    ControlParamType,
    Data,
    DwaParams,
    Dynamics,
    DynamicsParamType,
    InputType,
    LocationStateType,
    Slice,
    StateType,
    TwoDimArray,
    UnicycleControl,
    UnicycleStateType,
    VectorControl,
    VectorField,
)
from py_sim.vectorfield.vectorfields import G2GAvoid
from py_sim.worlds.polygon_world import PolygonWorld


class SingleAgentSim(Generic[StateType]):
    """Implements the main functions for a single agent simulation

    Attributes:
        params(SimParameters): parameters for running the simulation
        data(Data): Stores the current and next slice of information
        lock(Lock): Lock used for writing to the data
        stop(Event): Event used to indicate that the simulator should be stopped
        plots(PlotManifest): The manifest of plots to be plotted
    """
    def __init__(self,
                n_inputs: int,
                plots: PlotManifest[StateType],
                params: SimParameters[StateType]
                ) -> None:
        """Initialize the simulation
        """
        # Store Inputs
        self.params = params
        self.plots = plots

        # Create and store the data
        initial_slice: Slice[StateType] = Slice(state=self.params.initial_state, time=self.params.t0)
        self.data: Data[StateType] = Data(current=initial_slice)

        # Create a lock to store the data
        self.lock = Lock()

        # Create an event to stop the simulator
        self.stop = Event()

        # Initialize data storage
        self.initialize_data_storage(n_inputs=n_inputs)

    def update(self) -> None:
        """Performs all the required updates"""
        raise NotImplementedError("Update function must be implemented")

    def update_plot(self) -> None:
        """Plot the current values and state. Should be done with the lock on to avoid
           updating current while plotting the data
        """
        # Copy the state to avoid any conflicts
        with self.lock:
            plot_state = copy.deepcopy(self.data.current)

        # Update all of the state plotting elements
        for plotter in self.plots.state_plots:
            plotter.plot(state=plot_state.state)

        # Update all of the data plotting elements
        for plotter in self.plots.data_plots:
            plotter.plot(data=self.data)

        # Flush all of the figures
        for fig in self.plots.figs:
            fig.canvas.draw()
            fig.canvas.flush_events()

    def store_data_slice(self, sim_slice: Slice[StateType]) -> None:
        """Stores the state trajectory data

        Args:
            sim_slice: The information to be stored
        """
        with self.lock:
            # Check size - double if insufficient
            if self.data.traj_index_latest+1 >= self.data.state_traj.shape[1]: # Larger than allocated
                self.data.state_traj = np.append(self.data.state_traj, \
                    np.zeros(self.data.state_traj.shape), axis=1 )
                self.data.time_traj = np.append(self.data.time_traj, np.zeros(self.data.time_traj.size))
                self.data.control_traj = np.append(self.data.control_traj,
                                                   np.zeros(self.data.control_traj.shape),
                                                   axis=1)

            # Store data
            self.data.traj_index_latest += 1
            self.data.state_traj[:,self.data.traj_index_latest:self.data.traj_index_latest+1] = \
                sim_slice.state.state
            self.data.time_traj[self.data.traj_index_latest] = sim_slice.time

            if sim_slice.input_vec is not None:
                self.data.control_traj[:,self.data.traj_index_latest:self.data.traj_index_latest+1] = \
                sim_slice.input_vec

    def post_process(self) -> None:
        """Process the results"""
        print("Final state: ", self.data.current.state.state)
        print("State trajectory: ", self.data.state_traj)
        # print("Time trajectory: ", self.data.time_traj[0:self.data.traj_index_latest+1])

    def initialize_data_storage(self, n_inputs: int) -> None:
        """Initializes all of the storage

        Args:
            n_inputs: The number of inputs for the control trajectory
        """
        num_elements_traj: int = int( (self.params.tf - self.params.t0)/self.params.sim_step ) + 2
            # Number of elements in the trajectory + 2 for the start and end times
        self.data.state_traj = np.zeros((self.data.current.state.n_states, num_elements_traj))
        self.data.time_traj = np.zeros((num_elements_traj,))
        self.data.control_traj = np.zeros((n_inputs, num_elements_traj))
        self.data.traj_index_latest = -1 # -1 indicates that nothing has yet been saved

class SimpleSim(Generic[LocationStateType, InputType, ControlParamType, DynamicsParamType], SingleAgentSim[LocationStateType]):
    """Framework for implementing a simulator that just tests out a feedback controller

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType, DynamicsParamType]): The dynamics function to be used for simulation
        controller(Control[LocationStateType, InputType, DynamicsParamType, ControlParamType]): The control law to be used during simulation
        dynamic_params(DynamicsParamType): Fixed parameters for the dynamics
        control_params(ControlParamType): The parameters of the control law to be used in simulation
    """
    def __init__(self,
                dynamics: Dynamics[LocationStateType, InputType, DynamicsParamType],
                controller: Control[LocationStateType, InputType, DynamicsParamType, ControlParamType],
                dynamic_params: DynamicsParamType,
                control_params: ControlParamType,
                n_inputs: int,
                plots: PlotManifest[LocationStateType],
                params: SimParameters[LocationStateType]
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

            Args:
                initial_state: The starting state of the vehicle
                dynamics: The dynamics function to be used for simulation
                controller: The control law to be used during simulation
                dynamic_params: Fixed parameters for the dynamics
                control_params: The parameters of the control law to be used in simulation
                n_input: The number of inputs for the dynamics function
        """

        # Initialize the parent SingleAgentSim class
        super().__init__(n_inputs=n_inputs, plots=plots, params=params)

        # Initialize sim-specific parameters
        self.dynamics: Dynamics[LocationStateType, InputType, DynamicsParamType] = dynamics
        self.controller: Control[LocationStateType, InputType, DynamicsParamType, ControlParamType] = controller
        self.dynamic_params: DynamicsParamType = dynamic_params
        self.control_params: ControlParamType = control_params

    def update(self) -> None:
        """Calls all of the update functions

        The following are updated:
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """

        # Calculate the control
        control:InputType = self.controller(time=self.data.current.time,
                                state=self.data.current.state,
                                dyn_params=self.dynamic_params,
                                cont_params=self.control_params)
        self.data.current.input_vec = control.input

        # Update the state using the latest control
        self.data.next.state.state = euler_update(  dynamics=self.dynamics,
                                                    control=control,
                                                    initial=self.data.current.state,
                                                    params=self.dynamic_params,
                                                    dt=self.params.sim_step)

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

class VectorFollower(Generic[LocationStateType, InputType, ControlParamType, DynamicsParamType], SingleAgentSim[LocationStateType]):
    """Framework for implementing a simulator that uses a vector field for feedback

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType, DynamicsParamType]): The dynamics function to be used for simulation
        controller(VectorControl[LocationStateType, InputType, DynamicsParamType ControlParamType]): The control law to be used during simulation
        dynamic_params(DynamicsParamType): Fixed parameters for the dynamics
        control_params(ControlParamType): The parameters of the control law to be used in simulation
        vector_field(VectorField): Vector field that the vehicle will follow
    """
    def __init__(self,
                dynamics: Dynamics[LocationStateType, InputType, DynamicsParamType],
                controller: VectorControl[LocationStateType, InputType, DynamicsParamType, ControlParamType],
                dynamic_params: DynamicsParamType,
                control_params: ControlParamType,
                n_inputs: int,
                plots: PlotManifest[LocationStateType],
                vector_field: VectorField,
                params: SimParameters[LocationStateType]
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

        Args:
            initial_state: The starting state of the vehicle
            dynamics: The dynamics function to be used for simulation
            controller: The control law to be used during simulation
            dynamic_params: Fixed parameters for the dynamics
            control_params: The parameters of the control law to be used in simulation
            n_input: The number of inputs for the dynamics function
        """

        # Initialize the parent SingleAgentSim class
        super().__init__(n_inputs=n_inputs, plots=plots, params=params)

        # Initialize sim-specific parameters
        self.dynamics: Dynamics[LocationStateType, InputType, DynamicsParamType] = dynamics
        self.controller: VectorControl[LocationStateType, InputType, DynamicsParamType, ControlParamType] = controller
        self.dynamic_params: DynamicsParamType = dynamic_params
        self.control_params: ControlParamType = control_params
        self.vector_field: VectorField = vector_field

    def update(self) -> None:
        """Calls all of the updates in the sim.

          The following are updated:
            * Calculates the desired vector
            * Calculate the control to follow the vector
            * Update the state
            * Update the time
        """
        # Calculate the desired vector
        vec: TwoDimArray = self.vector_field.calculate_vector(state=self.data.current.state, time=self.data.current.time)

        # Calculate the control to follow the vector
        control:InputType = self.controller(time=self.data.current.time,
                                state=self.data.current.state,
                                vec=vec,
                                dyn_params=self.dynamic_params,
                                cont_params=self.control_params)
        self.data.current.input_vec = control.input

        # Update the state using the latest control
        self.data.next.state.state = euler_update(  dynamics=self.dynamics,
                                                    control=control,
                                                    initial=self.data.current.state,
                                                    params=self.dynamic_params,
                                                    dt=self.params.sim_step)

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

class NavFieldFollower(Generic[LocationStateType, InputType, ControlParamType, DynamicsParamType], SingleAgentSim[LocationStateType]):
    """Framework for implementing a simulator that uses a vector field for feedback control through a polygon world with a distance measurement

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType]): The dynamics function to be used for simulation
        controller(VectorControl[LocationStateType, InputType, DynamicsParamType, ControlParamType]): The control law to be used during simulation
        dynamic_params(DynamicsParamType): Fixed parameters for the dynamics
        control_params(ControlParamType): The parameters of the control law to be used in simulation
        vector_field(VectorField): Vector field that the vehicle will use to avoid obstacles while traversing to the goal
        world(PolygonWorld): World in which the vehicle is operating
        sensor(RangeBearingSensor): The sensor used for detecting obstacles
    """
    def __init__(self,  # pylint: disable=too-many-arguments
                dynamics: Dynamics[LocationStateType, InputType, DynamicsParamType],
                controller: VectorControl[LocationStateType, InputType, DynamicsParamType, ControlParamType],
                dynamic_params: DynamicsParamType,
                control_params: ControlParamType,
                n_inputs: int,
                plots: PlotManifest[LocationStateType],
                vector_field: VectorField,
                world: PolygonWorld,
                sensor: RangeBearingSensor,
                params: SimParameters[LocationStateType]
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

        Args:
            initial_state: The starting state of the vehicle
            dynamics: The dynamics function to be used for simulation
            controller: The control law to be used during simulation
            dynamic_params: Fixed parameters for the dynamics
            control_params: The parameters of the control law to be used in simulation
            n_input: The number of inputs for the dynamics function
            plots: The manifest of plots to be plotted
            vector_field: The field to follow
            world: The world through which the vehicle is navigating
            sensor: The range bearing sensor model
            params: Parameters controlling the simulation
        """

        # Initialize the parent SingleAgentSim class
        super().__init__(n_inputs=n_inputs, plots=plots, params=params)

        # Initialize sim-specific parameters
        self.dynamics: Dynamics[LocationStateType, InputType, DynamicsParamType] = dynamics
        self.controller: VectorControl[LocationStateType, InputType, DynamicsParamType, ControlParamType] = controller
        self.dynamic_params: DynamicsParamType = dynamic_params
        self.control_params: ControlParamType = control_params
        self.vector_field: VectorField = vector_field
        self.world: PolygonWorld = world
        self.sensor: RangeBearingSensor = sensor

    def update(self) -> None:
        """Calls all of the update functions.

        Updates performed:
            * Calculate the range and bearing measurements
            * Calculate the resulting vector to be followed
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """
        # Calculate the range bearing for the current position
        self.data.range_bearing_latest = self.sensor.calculate_range_bearing_measurement(\
            pose=self.data.current.state,
            world=self.world)

        # Calculate the desired vector
        vec: TwoDimArray = self.vector_field.calculate_vector(state=self.data.current.state, time=self.data.current.time)

        # Calculate the control to follow the vector
        control:InputType = self.controller(time=self.data.current.time,
                                state=self.data.current.state,
                                vec=vec,
                                dyn_params=self.dynamic_params,
                                cont_params=self.control_params)
        self.data.current.input_vec = control.input

        # Update the state using the latest control
        self.data.next.state.state = euler_update(  dynamics=self.dynamics,
                                                    control=control,
                                                    initial=self.data.current.state,
                                                    params=self.dynamic_params,
                                                    dt=self.params.sim_step)

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

class NavVectorFollower(Generic[LocationStateType, InputType, ControlParamType, DynamicsParamType], SingleAgentSim[LocationStateType]):
    """Framework for implementing a simulator that uses a vector field for feedback control through a polygon world with a distance measurement

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType, DynamicsParamType]): The dynamics function to be used for simulation
        controller(VectorControl[LocationStateType, InputType, DynamicsParamType, ControlParamType]): The control law to be used during simulation
        dynamic_params(DynamicsParamType): Fixed parameters for the dynamics
        control_params(ControlParamType): The parameters of the control law to be used in simulation
        vector_field(G2GAvoid): Vector field that the vehicle will use to avoid obstacles while traversing to the goal
        world(PolygonWorld): World in which the vehicle is operating
        sensor(RangeBearingSensor): The sensor used for detecting obstacles
        carrot(Optional[LineCarrot]): Provides a carrot to be followed
    """
    def __init__(self,  # pylint: disable=too-many-arguments
                dynamics: Dynamics[LocationStateType, InputType, DynamicsParamType],
                controller: VectorControl[LocationStateType, InputType, DynamicsParamType, ControlParamType],
                dynamic_params: DynamicsParamType,
                control_params: ControlParamType,
                n_inputs: int,
                plots: PlotManifest[LocationStateType],
                vector_field: G2GAvoid,
                world: PolygonWorld,
                sensor: RangeBearingSensor,
                carrot: Optional[LineCarrot],
                params: SimParameters[LocationStateType]
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

        Args:
            initial_state: The starting state of the vehicle
            dynamics: The dynamics function to be used for simulation
            controller: The control law to be used during simulation
            dynamic_params: Fixed parameters for the dynamics
            control_params: The parameters of the control law to be used in simulation
            n_input: The number of inputs for the dynamics function
        """

        # Initialize the parent SingleAgentSim class
        super().__init__(n_inputs=n_inputs, plots=plots, params=params)

        # Initialize sim-specific parameters
        self.dynamics: Dynamics[LocationStateType, InputType, DynamicsParamType] = dynamics
        self.controller: VectorControl[LocationStateType, InputType, DynamicsParamType, ControlParamType] = controller
        self.dynamic_params: DynamicsParamType = dynamic_params
        self.control_params: ControlParamType = control_params
        self.vector_field: G2GAvoid = vector_field
        self.world: PolygonWorld = world
        self.sensor: RangeBearingSensor = sensor
        self.carrot: Optional[LineCarrot] = carrot

    def update(self) -> None:
        """Calls all of the update functions.

        The following updates are performed:
            * Calculate the range and bearing measurements
            * Update the goal location if carrot-following
            * Calculate the resulting vector to be followed
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """
        self.update_sensing()
        self.update_dynamics()

    def update_sensing(self) -> None:
        """Updates the range and bearing measurements
        """
        # Copy of the current state
        with self.lock:
            pose = copy.deepcopy(self.data.current.state)

        # Calculate the range bearing for the current position
        self.data.range_bearing_latest = self.sensor.calculate_range_bearing_measurement(\
            pose=pose,
            world=self.world)

    def update_dynamics(self) -> None:
        """Updates the control and dynamics

        The following updates are performed:
            * Update the goal location if carrot-following
            * Calculate the resulting vector to be followed
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """
        # Update the goal position
        if self.carrot is not None:
            self.vector_field.x_g = \
                self.carrot.get_carrot_point(point=self.data.current.state)

        # Calculate the desired vector
        try:
            self.vector_field.update_obstacles(locations=self.data.range_bearing_latest.location)
        except ValueError:
            print("Warning: the input locations have the wrong number of locations. Obstacles not updated")
        vec: TwoDimArray = self.vector_field.calculate_vector(state=self.data.current.state, time=self.data.current.time)

        # Calculate the control to follow the vector
        control:InputType = self.controller(time=self.data.current.time,
                                state=self.data.current.state,
                                vec=vec,
                                dyn_params=self.dynamic_params,
                                cont_params=self.control_params)
        self.data.current.input_vec = control.input

        # Update the state using the latest control
        self.data.next.state.state = euler_update(  dynamics=self.dynamics,
                                                    control=control,
                                                    initial=self.data.current.state,
                                                    params=self.dynamic_params,
                                                    dt=self.params.sim_step)

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

class DwaFollower(Generic[UnicycleStateType, InputType, DynamicsParamType], SingleAgentSim[UnicycleStateType]):
    """Framework for implementing a simulator that uses a vector field for feedback control through a polygon world with a distance measurement

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType, DynamicsParamType]): The dynamics function to be used for simulation
        controller(Control[UnicycleStateType, InputType, DynamicsParamType, ControlParamType]): The control law to be used during simulation
        dynamic_params(DynamicsParamType): Fixed parameters for the dynamics
        control_params(ArcParams): The parameters of the control law to be used in simulation
        vector_field(G2GAvoid): Vector field that the vehicle will use to avoid obstacles while traversing to the goal
        world(PolygonWorld): World in which the vehicle is operating
        sensor(RangeBearingSensor): The sensor used for detecting obstacles
        carrot(LineCarrot): Provides a carrot to be followed
        dwa_params(DwaParams): Parameters for the DWA algorithm
        dwa_arc(UnicycleControl): The arc parameters for the DWA algorithm

        arc_plot(sim_plt.ControlArcPlot): The plot for the chosen arc of the DWA algorithm
    """
    def __init__(self,  # pylint: disable=too-many-arguments
                 dynamics: Dynamics[UnicycleStateType, InputType, DynamicsParamType],
                 controller: Control[UnicycleStateType, InputType, DynamicsParamType, ArcParams],
                 dynamic_params: DynamicsParamType,
                 dwa_params: DwaParams,
                 n_inputs: int,
                 plots: PlotManifest[UnicycleStateType],
                 world: dwa.World,
                 carrot: LineCarrot,
                 params: SimParameters[UnicycleStateType]
                 ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

        Args:
            initial_state: The starting state of the vehicle
            dynamics: The dynamics function to be used for simulation
            controller: The control law to be used during simulation
            dynamic_params: Fixed parameters for the dynamics
            control_params: The parameters of the control law to be used in simulation
            n_input: The number of inputs for the dynamics function
        """

        # Initialize the parent SingleAgentSim class
        super().__init__(n_inputs=n_inputs, plots=plots, params=params)

        # Initialize sim-specific parameters
        self.dynamics: Dynamics[UnicycleStateType, InputType, DynamicsParamType] = dynamics
        self.controller: Control[UnicycleStateType, InputType, DynamicsParamType, ArcParams] = controller
        self.dynamic_params: DynamicsParamType = dynamic_params
        self.world: dwa.World = world
        self.carrot: LineCarrot = carrot

        # Store the DWA values
        self.dwa_params: DwaParams = dwa_params
        self.dwa_arc: UnicycleControl = UnicycleControl(v=0., w=0.)

        # Initialize the dwa specific plots
        self.arc_plot: sim_plt.ControlArcPlot[UnicycleStateType] = sim_plt.ControlArcPlot(ax=self.plots.vehicle_axes)

    def update(self) -> None:
        """Calls all of the update functions.

        The following updates are performed:
            * Update the goal location if carrot-following
            * Calculate the DWA arc parameters
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """
        # Update the goal position
        x_g = self.carrot.get_carrot_point(point=self.data.current.state)

        # Calculate the DWA arc parameters
        self.dwa_arc = dwa.compute_desired_velocities(
            state=self.data.current.state,
            params=self.dwa_params,
            goal=x_g,
            world=self.world)

        # Calculate the control to follow the vector
        arc_params = ArcParams(v_d=self.dwa_arc.v, w_d = self.dwa_arc.w)
        control:InputType = self.controller(time=self.data.current.time,
                                            state=self.data.current.state,
                                            dyn_params=self.dynamic_params,
                                            cont_params=arc_params)
        self.data.current.input_vec = control.input

        # Update the state using the latest control
        self.data.next.state.state = euler_update(dynamics=self.dynamics,
                                                  control=control,
                                                  initial=self.data.current.state,
                                                  params=self.dynamic_params,
                                                  dt=self.params.sim_step)

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

    def update_plot(self) -> None:
        """Plot the default plots as well as the dwa plots
        """
        # Plot the dwa plots
        self.arc_plot.plot(state=self.data.current.state,
                           control=self.dwa_arc,
                           ds=self.dwa_params.ds,
                           tf=self.dwa_params.tf)

        # Plot the environment and plan
        super().update_plot()
