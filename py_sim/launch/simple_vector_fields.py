"""simple_vector_fields.py: Provides a series of examples of vector field following for basic fields
"""

from typing import Generic

from py_sim.dynamics.unicycle import UniVelVecParams
from py_sim.dynamics.unicycle import dynamics as unicycle_dynamics
from py_sim.dynamics.unicycle import velocityVectorFieldControl
from py_sim.sim.generic_sim import SingleAgentSim, start_simple_sim
from py_sim.sim.integration import euler_update
from py_sim.tools.plot_constructor import create_plot_manifest
from py_sim.tools.plotting import PlotManifest
from py_sim.tools.sim_types import (
    ControlParamType,
    Dynamics,
    InputType,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
    UnicycleStateType,
    VectorControl,
    VectorField,
)
from py_sim.vectorfield.vectorfields import (  # pylint: disable=unused-import
    AvoidObstacle,
    GoToGoalField,
    SummedField,
)


class VectorFollower(Generic[UnicycleStateType, InputType, ControlParamType], SingleAgentSim[UnicycleStateType]):
    """Framework for implementing a simulator that just tests out a feedback controller"""
    def __init__(self,
                initial_state: UnicycleStateType,
                dynamics: Dynamics[UnicycleStateType, InputType],
                controller: VectorControl[UnicycleStateType, InputType, ControlParamType],
                control_params: ControlParamType,
                n_inputs: int,
                plots: PlotManifest[UnicycleStateType],
                vector_field: VectorField
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

            Inputs:
                initial_state: The starting state of the vehicle
                dynamics: The dynamics function to be used for simulation
                controller: The control law to be used during simulation
                control_params: The parameters of the control law to be used in simulation
                n_input: The number of inputs for the dynamics function
        """

        super().__init__(initial_state=initial_state, n_inputs=n_inputs, plots=plots)

        # Initialize sim-specific parameters
        self.dynamics: Dynamics[UnicycleStateType, InputType] = dynamics
        self.controller: VectorControl[UnicycleStateType, InputType, ControlParamType] = controller
        self.control_params: ControlParamType = control_params
        self.vector_field: VectorField = vector_field

    def update(self) -> None:
        """Calls all of the update functions
            * Calculate the control to be executed
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

def run_simple_vectorfield_example() -> None:
    """Runs an example of a go-to-goal vector field"""
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
                                 unicycle_time_series=True,
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

if __name__ == "__main__":
    run_simple_vectorfield_example()
