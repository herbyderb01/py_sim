"""simple_sim.py performs a test with a single vehicle

    Provides an example of using a control law within the simulation with position and state plotting occurring actively during the movement of the vehicle.

"""

from typing import Generic

from py_sim.dynamics import single_integrator
from py_sim.dynamics.unicycle import arc_control
from py_sim.dynamics.unicycle import dynamics as unicycle_dynamics
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.plotting.plotting import PlotManifest
from py_sim.sim.generic_sim import SingleAgentSim, start_simple_sim
from py_sim.sim.integration import euler_update
from py_sim.tools.sim_types import (
    ArcParams,
    Control,
    ControlParamType,
    Dynamics,
    InputType,
    LocationStateType,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
)


class SimpleSim(Generic[LocationStateType, InputType, ControlParamType], SingleAgentSim[LocationStateType]):
    """Framework for implementing a simulator that just tests out a feedback controller

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType]): The dynamics function to be used for simulation
        controller(Control[LocationStateType, InputType, ControlParamType]): The control law to be used during simulation
        control_params(ControlParamType): The parameters of the control law to be used in simulation

    """
    def __init__(self,
                initial_state: LocationStateType,
                dynamics: Dynamics[LocationStateType, InputType],
                controller: Control[LocationStateType, InputType, ControlParamType],
                control_params: ControlParamType,
                n_inputs: int,
                plots: PlotManifest[LocationStateType]
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
        self.controller: Control[LocationStateType, InputType, ControlParamType] = controller
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
                                params=self.control_params)
        self.data.current.input_vec = control.input

        # Update the state using the latest control
        self.data.next.state.state = euler_update(  dynamics=self.dynamics,
                                                    control=control,
                                                    initial=self.data.current.state,
                                                    dt=self.params.sim_step)

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

def run_unicycle_arc_example() -> None:
    """Runs an example of a vehicle executing an arc"""
    # Initialize the state and control
    arc_params = ArcParams(v_d=1., w_d= 1.)
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-2, 2),
                                 x_limits=(-2, 2),
                                 position_dot=False,
                                 position_triangle=True,
                                 state_trajectory=True,
                                 time_series=True)

    # Create the simulation
    sim = SimpleSim(initial_state=state_initial,
                    dynamics=unicycle_dynamics,
                    controller=arc_control,
                    control_params=arc_params,
                    n_inputs=UnicycleControl.n_inputs,
                    plots=plot_manifest)

    # Update the simulation step variables
    sim.params.sim_plot_period = 0.2
    sim.params.sim_step = 0.1
    sim.params.sim_update_period = 0.01
    start_simple_sim(sim=sim)

def run_integrator_example() -> None:
    """Runs an example of a single integrator executing a straight line"""
    # Initialize the state and control
    const_params = single_integrator.ConstantInputParams(v_d=TwoDimArray(x=1., y=1.))
    state_initial = TwoDimArray(x = 0., y= 0.)

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-2, 2),
                                 x_limits=(-2, 2),
                                 position_dot=True,
                                 state_trajectory=True,
                                 time_series=True)

    # Create the simulation
    sim = SimpleSim(initial_state=state_initial,
                    dynamics=single_integrator.dynamics,
                    controller=single_integrator.const_control,
                    control_params=const_params,
                    n_inputs=single_integrator.PointInput.n_inputs,
                    plots=plot_manifest)

    # Update the simulation step variables
    sim.params.sim_plot_period = 0.2
    sim.params.sim_step = 0.1
    sim.params.sim_update_period = 0.01
    start_simple_sim(sim=sim)


if __name__ == "__main__":
    #run_unicycle_arc_example()
    run_integrator_example()
