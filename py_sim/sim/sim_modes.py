""" Defines a number of simulation modes that can be used
"""

from typing import Generic, Optional

from py_sim.plotting.plotting import PlotManifest
from py_sim.sensors.range_bearing import RangeBearingSensor
from py_sim.sim.generic_sim import SimParameters, SingleAgentSim
from py_sim.sim.integration import euler_update
from py_sim.tools.projections import LineCarrot
from py_sim.tools.sim_types import (
    Control,
    ControlParamType,
    Dynamics,
    DynamicsParamType,
    InputType,
    LocationStateType,
    TwoDimArray,
    VectorControl,
    VectorField,
)
from py_sim.vectorfield.vectorfields import G2GAvoid
from py_sim.worlds.polygon_world import PolygonWorld


class SimpleSim(Generic[LocationStateType, InputType, ControlParamType, DynamicsParamType], SingleAgentSim[LocationStateType]):
    """Framework for implementing a simulator that just tests out a feedback controller

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType, DynamicsParamType]): The dynamics function to be used for simulation
        controller(Control[LocationStateType, InputType, ControlParamType]): The control law to be used during simulation
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
        controller(Control[LocationStateType, InputType, ControlParamType]): The control law to be used during simulation
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

class NavVectorFollower(Generic[LocationStateType, InputType, ControlParamType, DynamicsParamType], SingleAgentSim[LocationStateType]):
    """Framework for implementing a simulator that uses a vector field for feedback control through a polygon world with a distance measurement

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType, DynamicsParamType]): The dynamics function to be used for simulation
        controller(Control[LocationStateType, InputType, ControlParamType]): The control law to be used during simulation
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

        Updates performed:
            * Calculate the range and bearing measurements
            * Update the goal location if carrot-following
            * Calculate the resulting vector to be followed
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """
        # Calculate the range bearing for the current position
        self.data.range_bearing_latest = self.sensor.calculate_range_bearing_measurement(\
            pose=self.data.current.state,
            world=self.world)

        # Update the goal position
        if self.carrot is not None:
            self.vector_field.x_g = \
                self.carrot.get_carrot_point(point=self.data.current.state)

        # Calculate the desired vector
        self.vector_field.update_obstacles(locations=self.data.range_bearing_latest.location)
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
        controller(Control[LocationStateType, InputType, ControlParamType]): The control law to be used during simulation
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
