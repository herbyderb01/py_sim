"""simple_sim.py performs a test with a single vehicle

    Provides an example of using a control law within the simulation with position and state plotting occurring actively during the movement of the vehicle.

"""
import py_sim.dynamics.bicycle as bike # pylint: disable=unused-import
import py_sim.dynamics.differential_drive as diff # pylint: disable=unused-import
import py_sim.dynamics.unicycle as uni # pylint: disable=unused-import
from py_sim.dynamics import single_integrator
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.sim.generic_sim import SimParameters, start_simple_sim
from py_sim.sim.sim_modes import SimpleSim
from py_sim.tools.sim_types import (
    ArcParams,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
)


def run_simple_arc_example() -> None:
    """Runs an example of a vehicle executing an arc

    Args:
        model: The model used for the simple dynamics
    """
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
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.2
    params.sim_step = 0.1
    params.sim_update_period = 0.01
    sim = SimpleSim(params=params,
                    # dynamics=uni.dynamics,
                    # controller=uni.arc_control,
                    # dynamic_params= uni.UnicycleParams(),
                    # dynamics=diff.dynamics,
                    # controller=diff.arc_control,
                    # dynamic_params=diff.DiffDriveParams(L = 0.25, R=0.025),
                    dynamics=bike.dynamics,
                    controller=bike.arc_control,
                    dynamic_params=bike.BicycleParams(L = 1.),
                    control_params=arc_params,
                    n_inputs=UnicycleControl.n_inputs,
                    plots=plot_manifest)

    # Run the simulation
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
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.2
    params.sim_step = 0.1
    params.sim_update_period = 0.01
    sim = SimpleSim(params=params,
                    dynamics=single_integrator.dynamics,
                    controller=single_integrator.const_control,
                    dynamic_params=single_integrator.SingleIntegratorParams(),
                    control_params=const_params,
                    n_inputs=single_integrator.PointInput.n_inputs,
                    plots=plot_manifest)

    # Run the simulation
    start_simple_sim(sim=sim)

if __name__ == "__main__":
    #run_integrator_example()
    run_simple_arc_example()
