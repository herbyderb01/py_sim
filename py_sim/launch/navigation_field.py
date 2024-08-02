"""vector_field_nav.py: Provides sample vector fields used for navigation
"""

import py_sim.dynamics.unicycle as uni
from py_sim.dynamics import single_integrator
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.sensors.range_bearing import RangeBearingSensor
from py_sim.sim.generic_sim import SimParameters, start_sim
from py_sim.sim.sim_modes import NavFieldFollower
from py_sim.tools.sim_types import TwoDimArray, UnicycleControl, UnicycleState
from py_sim.vectorfield.grid_navigation_function import GridNavigationFunction
from py_sim.worlds.polygon_world import (
    generate_non_convex_obstacles,
    generate_world_obstacles,
)


def run_unicycle_simple_vectorfield_example() -> None:
    """Runs an example of a go-to-goal vector field combined with obstacle avoidance to show off the sensor measurements being performed
    """

    # Initialize the state and control
    vel_params = uni.UniVelVecParams(vd_field_max=5., k_wd= 5.)
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)

    # Set the number of sensor lines
    n_lines = 10

    # Create the obstacle world
    obstacle_world = generate_world_obstacles()
    #obstacle_world = generate_non_convex_obstacles()

    # Create the navigation vector field
    nav_field = GridNavigationFunction(end=TwoDimArray(x=10., y=5.),
                                       obstacle_world=obstacle_world,
                                       plan_type="dijkstra",
                                       v_des=vel_params.vd_field_max,
                                       sig=1.,
                                       x_lim = (-5., 25.),
                                       y_lim = (-5., 10.))

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 position_dot=False,
                                 position_triangle=True,
                                 state_trajectory=True,
                                 time_series=True,
                                 vectorfield=nav_field,
                                 vector_res=0.5,
                                 world=obstacle_world,
                                 range_bearing_locations=True,
                                 range_bearing_lines=True
                                 )

    # Create the simulation
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.1
    params.sim_step = 0.1
    params.sim_update_period = 0.1
    sim = NavFieldFollower(params=params,
                           dynamics= uni.dynamics,
                           controller=uni.velocity_vector_field_control,
                           dynamic_params= uni.UnicycleParams(),
                           control_params=vel_params,
                           n_inputs=UnicycleControl.n_inputs,
                           plots=plot_manifest,
                           vector_field=nav_field,
                           world=obstacle_world,
                           sensor=RangeBearingSensor(n_lines=n_lines, max_dist=4.),
                         )

    # Run the simulation
    start_sim(sim=sim)

def run_single_simple_vectorfield_example() -> None:
    """Runs an example of a go-to-goal vector field combined with obstacle avoidance to show off the sensor measurements being performed using a single integrator
    """
    # Initialize the state and control
    vel_params = single_integrator.VectorParams(v_max=5.)
    state_initial = TwoDimArray(x = 0., y= 0.)

    # Set the number of sensor lines
    n_lines = 10

    # Create the obstacle world
    #obstacle_world = generate_world_obstacles()
    obstacle_world = generate_non_convex_obstacles()

    # Create the navigation vector field
    nav_field = GridNavigationFunction(end=TwoDimArray(x=10., y=5.),
                                       obstacle_world=obstacle_world,
                                       plan_type="dijkstra",
                                       v_des=vel_params.v_max,
                                       sig=1.,
                                       x_lim = (-5., 25.),
                                       y_lim = (-5., 10.))

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 position_dot=True,
                                 state_trajectory=True,
                                 time_series=True,
                                 vectorfield=nav_field,
                                 vector_res=0.5,
                                 world=obstacle_world,
                                 range_bearing_locations=True,
                                 range_bearing_lines=True
                                 )

    # Create the simulation
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.1
    params.sim_step = 0.01
    params.sim_update_period = 0.01
    sim = NavFieldFollower(params=params,
                           dynamics=single_integrator.dynamics,
                           controller=single_integrator.vector_control,
                           dynamic_params=single_integrator.SingleIntegratorParams(),
                           control_params=vel_params,
                           n_inputs=single_integrator.PointInput.n_inputs,
                           plots=plot_manifest,
                           vector_field=nav_field,
                           world=obstacle_world,
                           sensor=RangeBearingSensor(n_lines=n_lines, max_dist=4.)
                         )

    # Run the simulation
    start_sim(sim=sim)

if __name__ == "__main__":
    # Perform navigation without path planning (simple goal and avoid vector fields)
    #run_single_simple_vectorfield_example()
    run_unicycle_simple_vectorfield_example()
