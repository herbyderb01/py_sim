"""simple_vector_fields.py: Provides a series of examples of vector field following for basic fields

Provides and example of using a vector field control law with the vector field overlaid onto the live plotting of the moving vehicle
"""

import numpy as np
import py_sim.vectorfield.vectorfields as vf
from py_sim.dynamics import single_integrator
from py_sim.launch.navigation_field import NavFieldFollower
from py_sim.launch.simple_vector_fields import VectorFollower
from py_sim.launch.vector_field_nav import NavVectorFollower
from py_sim.path_planning.path_generation import create_path
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.sensors.range_bearing import RangeBearingSensor
from py_sim.sim.generic_sim import SimParameters, start_simple_sim
from py_sim.tools.projections import LineCarrot
from py_sim.tools.sim_types import TwoDimArray
from py_sim.vectorfield.grid_navigation_function import GridNavigationFunction
from py_sim.worlds.polygon_world import (  # pylint: disable=unused-import
    generate_non_convex_obstacles,
    generate_world_obstacles,
)


def simple_vectorfield() -> None:
    """Runs an example of a go-to-goal vector field combined with an obstacle avoidance
    field.
    """
    # Initialize the state and control
    vel_params = single_integrator.VectorParams(v_max=5.)
    state_initial = TwoDimArray(x = 0., y= 0.)

    # Create the vector field
    vector_field_g2g = vf.GoToGoalField(x_g=TwoDimArray(x=-4., y=2.),
                                        v_max=vel_params.v_max,
                                        sig=1.)
    # vector_field_avoid = vf.AvoidObstacle(x_o=TwoDimArray(x=0., y=1.),
    #                                       v_max=vel_params.v_max,
    #                                       S=2.,
    #                                       R=1.)
    # vector_field = vf.SummedField(fields=[vector_field_g2g, vector_field_avoid],
    #                               weights=[1., 1.],
    #                               v_max=vel_params.v_max)
    # vector_field =vector_field_avoid
    vector_field = vector_field_g2g


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
                         n_inputs=single_integrator.PointInput.n_inputs,
                         plots=plot_manifest,
                         vector_field=vector_field)

    # Run the simulation
    start_simple_sim(sim=sim)

def carrot_follow() -> None:
    """Runs an example of a go-to-goal vector field combined with obstacle avoidance to show off the sensor measurements being performed using a single integrator

    Args:
        follow_path: True => a path will be created and followed, False => the vector field will alone be used
                     for navigating to the goal
    """
    # Initialize the state and control
    vel_params = single_integrator.VectorParams(v_max=5.)
    state_initial = TwoDimArray(x = 0., y= 0.)

    # Create the vector field
    n_lines = 10 # Number of sensor lines
    vector_field = vf.G2GAvoid(x_g=TwoDimArray(x=8., y=5.),
                               n_obs=n_lines,
                               v_max=vel_params.v_max,
                               S=1.5,
                               R=1.,
                               sig=1.)

    # Create the obstacle world
    #obstacle_world = generate_world_obstacles()
    obstacle_world = generate_non_convex_obstacles()

    # Create the plan
    plan = create_path(start=TwoDimArray(x=state_initial.x, y=state_initial.y), end=vector_field.x_g, obstacle_world=obstacle_world, plan_type="voronoi")
    if plan is not None:
        line = np.array([plan[0], plan[1]])
        carrot = LineCarrot(line=line, s_dev_max=5., s_carrot=2.)
    else:
        carrot = None


    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 position_dot=True,
                                 state_trajectory=True,
                                 time_series=True,
                                 vectorfield=vector_field,
                                 vector_res=0.5,
                                 world=obstacle_world,
                                 range_bearing_locations=True,
                                 range_bearing_lines=True,
                                 plan=plan,
                                 line_carrot=carrot)

    # Create the simulation
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.1
    params.sim_step = 0.1
    params.sim_update_period = 0.1
    params.tf = 5.
    sim = NavVectorFollower(params=params,
                            dynamics=single_integrator.dynamics,
                            controller=single_integrator.vector_control,
                            control_params=vel_params,
                            n_inputs=single_integrator.PointInput.n_inputs,
                            plots=plot_manifest,
                            vector_field=vector_field,
                            world=obstacle_world,
                            sensor=RangeBearingSensor(n_lines=n_lines, max_dist=4.),
                            carrot=carrot
                         )

    # Run the simulation
    start_simple_sim(sim=sim)

def navigation_function() -> None:
    """Runs an example of a go-to-goal vector field combined with obstacle avoidance to show off the sensor measurements being performed using a single integrator
    """
    # Initialize the state and control
    vel_params = single_integrator.VectorParams(v_max=5.)
    state_initial = TwoDimArray(x = 0., y= 0.)

    # Create the sensors
    n_lines = 10 # Number of sensor lines

    # Create the obstacle world
    obstacle_world = generate_world_obstacles()
    #obstacle_world = generate_non_convex_obstacles()

    # Create the navigation vector field
    nav_field = GridNavigationFunction(end=TwoDimArray(x=8., y=5.),
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
    params.tf = 5.
    sim = NavFieldFollower(params=params,
                           dynamics=single_integrator.dynamics,
                           controller=single_integrator.vector_control,
                           control_params=vel_params,
                           n_inputs=single_integrator.PointInput.n_inputs,
                           plots=plot_manifest,
                           vector_field=nav_field,
                           world=obstacle_world,
                           sensor=RangeBearingSensor(n_lines=n_lines, max_dist=4.)
                         )

    # Run the simulation
    start_simple_sim(sim=sim)

if __name__ == "__main__":
    #simple_vectorfield()
    #carrot_follow()
    navigation_function()
