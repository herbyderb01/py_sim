"""simple_vector_fields.py: Provides a series of examples of vector field following for basic fields

Provides and example of using a vector field control law with the vector field overlaid onto the live plotting of the moving vehicle
"""

from py_sim.dynamics import single_integrator
from py_sim.dynamics import unicycle as uni
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.sim.generic_sim import SimParameters, start_simple_sim
from py_sim.sim.sim_modes import VectorFollower
from py_sim.tools.sim_types import TwoDimArray, UnicycleControl, UnicycleState
from py_sim.vectorfield.vectorfields import (  # pylint: disable=unused-import
    AvoidObstacle,
    GoToGoalField,
    SummedField,
)


def run_unicycle_simple_vectorfield_example() -> None:
    """Runs an example of a go-to-goal vector field combined with an obstacle avoidance
    field.
    """
    # Initialize the state and control
    vel_params = uni.UniVelVecParams(vd_field_max=5., k_wd= 2.)
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
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.2
    params.sim_step = 0.1
    params.sim_update_period = 0.01
    params.tf = 5.
    sim = VectorFollower(params=params,
                         dynamics=uni.dynamics,
                         controller=uni.velocity_vector_field_control,
                         dynamic_params=uni.UnicycleParams(),
                         control_params=vel_params,
                         n_inputs=UnicycleControl.n_inputs,
                         plots=plot_manifest,
                         vector_field=vector_field)

    # Run the simulation
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
    # vector_field =vector_field_avoid
    # vector_field = vector_field_g2g


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
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.1
    params.sim_step = 0.01
    params.sim_update_period = 0.01
    params.tf = 5.
    sim = VectorFollower(params=params,
                         dynamics=single_integrator.dynamics,
                         controller=single_integrator.vector_control,
                         control_params=vel_params,
                         dynamic_params=single_integrator.SingleIntegratorParams(),
                         n_inputs=single_integrator.PointInput.n_inputs,
                         plots=plot_manifest,
                         vector_field=vector_field)

    # Run the simulation
    start_simple_sim(sim=sim)

if __name__ == "__main__":
    run_single_integrator_simple_vectorfield_example()
    #run_unicycle_simple_vectorfield_example()
