"""simple_vector_fields.py: Provides a series of examples of vector field following for basic fields

Provides and example of using a vector field control law with the vector field overlaid onto the live plotting of the moving vehicle
"""

from typing import Generic

from py_sim.dynamics import single_integrator
from py_sim.dynamics.unicycle import UniVelVecParams
from py_sim.dynamics.unicycle import dynamics as unicycle_dynamics
from py_sim.dynamics.unicycle import velocityVectorFieldControl
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.plotting.plotting import PlotManifest
from py_sim.sim.generic_sim import SingleAgentSim, start_simple_sim
from py_sim.sim.integration import euler_update
from py_sim.tools.sim_types import (
    ControlParamType,
    Dynamics,
    InputType,
    LocationStateType,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
    VectorControl,
    VectorField,
)
from py_sim.vectorfield.vectorfields import (  # pylint: disable=unused-import
    AvoidObstacle,
    GoToGoalField,
    SummedField,
)


class VectorFollower(Generic[LocationStateType, InputType, ControlParamType], SingleAgentSim[LocationStateType]):
    """Framework for implementing a simulator that uses a vector field for feedback

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType]): The dynamics function to be used for simulation
        controller(Control[LocationStateType, InputType, ControlParamType]): The control law to be used during simulation
        control_params(ControlParamType): The parameters of the control law to be used in simulation
        vector_field(VectorField): Vector field that the vehicle will follow
    """
    def __init__(self,
                initial_state: LocationStateType,
                dynamics: Dynamics[LocationStateType, InputType],
                controller: VectorControl[LocationStateType, InputType, ControlParamType],
                control_params: ControlParamType,
                n_inputs: int,
                plots: PlotManifest[LocationStateType],
                vector_field: VectorField
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

        Args:
            initial_state: The starting state of the vehicle
            dynamics: The dynamics function to be used for simulation
            controller: The control law to be used during simulation
            control_params: The parameters of the control law to be used in simulation
            n_input: The number of inputs for the dynamics function
        """

        super().__init__(initial_state=initial_state, n_inputs=n_inputs, plots=plots)

        # Initialize sim-specific parameters
        self.dynamics: Dynamics[LocationStateType, InputType] = dynamics
        self.controller: VectorControl[LocationStateType, InputType, ControlParamType] = controller
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
                                params=self.control_params)
        self.data.current.input_vec = control.input

        # Update the state using the latest control
        self.data.next.state.state = euler_update(  dynamics=self.dynamics,
                                                    control=control,
                                                    initial=self.data.current.state,
                                                    dt=self.params.sim_step)

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

def run_unicycle_simple_vectorfield_example() -> None:
    """Runs an example of a go-to-goal vector field combined with an obstacle avoidance
    field.
    """
    # Initialize the state and control
    vel_params = UniVelVecParams(vd_field_max=5., k_wd= 2.)
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)

    # Create the vector field
    vector_field_g2g = GoToGoalField(x_g=TwoDimArray(x=-4., y=2.), v_max=vel_params.vd_field_max, sig=1)
    vector_field_avoid = AvoidObstacle(x_o=TwoDimArray(x=0., y=1.), v_max=vel_params.vd_field_max, S=2., R=1.)
    vector_field = SummedField(fields=[vector_field_g2g, vector_field_avoid],
                               weights=[1., 1.],
                               v_max=vel_params.vd_field_max)
    #vector_field = vector_field_g2g

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 5),
                                 x_limits=(-5, 5),
                                 position_dot=False,
                                 position_triangle=True,
                                 state_trajectory=True,
                                 time_series=True,
                                 vectorfield=vector_field,
                                 vector_res=0.4)

    # Create the simulation
    sim = VectorFollower(initial_state=state_initial,
                         dynamics=unicycle_dynamics,
                         controller=velocityVectorFieldControl,
                         control_params=vel_params,
                         n_inputs=UnicycleControl.n_inputs,
                         plots=plot_manifest,
                         vector_field=vector_field)

    # Update the simulation step variables
    sim.params.sim_plot_period = 0.2
    sim.params.sim_step = 0.1
    sim.params.sim_update_period = 0.01
    start_simple_sim(sim=sim)

def run_single_integrator_simple_vectorfield_example() -> None:
    """Runs an example of a go-to-goal vector field combined with an obstacle avoidance
    field.
    """
    # Initialize the state and control
    vel_params = single_integrator.VectorParams(v_max=5.)
    state_initial = TwoDimArray(x = 0., y= 0.)

    # Create the vector field
    vector_field_g2g = GoToGoalField(x_g=TwoDimArray(x=-4., y=2.), v_max=vel_params.v_max, sig=1)
    vector_field_avoid = AvoidObstacle(x_o=TwoDimArray(x=0., y=1.), v_max=vel_params.v_max, S=2., R=1.)
    vector_field = SummedField(fields=[vector_field_g2g, vector_field_avoid],
                               weights=[1., 1.],
                               v_max=vel_params.v_max)
    #vector_field = vector_field_g2g

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 5),
                                 x_limits=(-5, 5),
                                 position_dot=True,
                                 state_trajectory=True,
                                 time_series=True,
                                 vectorfield=vector_field,
                                 vector_res=0.4)

    # Create the simulation
    sim = VectorFollower(initial_state=state_initial,
                         dynamics=single_integrator.dynamics,
                         controller=single_integrator.vector_control,
                         control_params=vel_params,
                         n_inputs=single_integrator.PointInput.n_inputs,
                         plots=plot_manifest,
                         vector_field=vector_field)

    # Update the simulation step variables
    sim.params.sim_plot_period = 0.2
    sim.params.sim_step = 0.1
    sim.params.sim_update_period = 0.01
    start_simple_sim(sim=sim)

if __name__ == "__main__":
    #run_unicycle_simple_vectorfield_example()
    run_single_integrator_simple_vectorfield_example()
