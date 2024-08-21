"""03_Vehicle_models.py contains functions required for running homework #3
"""
import matplotlib.pyplot as plt
import numpy as np
import py_sim.dynamics.bicycle as bike
import py_sim.dynamics.differential_drive as diff
import py_sim.dynamics.unicycle as uni
import py_sim.sensors.occupancy_grid as og
import py_sim.worlds.polygon_world as poly_world
from matplotlib.axes import Axes
from py_sim.dynamics.unicycle import solution_trajectory as unicycle_solution_trajectory
from py_sim.path_planning import dwa
from py_sim.path_planning.path_generation import create_path
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.sensors.range_bearing import RangeBearingSensor
from py_sim.sim.generic_sim import SimParameters, start_sim
from py_sim.sim.sim_modes import (
    DwaFollower,
    NavVectorFollower,
    SimpleSim,
    VectorFollower,
)
from py_sim.tools.projections import LineCarrot
from py_sim.tools.sim_types import (
    ArcParams,
    DwaParams,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
)
from py_sim.vectorfield.vectorfields import (  # pylint: disable=unused-import
    AvoidObstacle,
    G2GAvoid,
    GoToGoalField,
    SummedField,
)


def run_unicycle_arc_example() -> None:
    """Runs an example of a unicycle vehicle executing an arc

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
    params.tf = 5.
    sim = SimpleSim(params=params,
                    dynamics=uni.dynamics,  # Unicycle
                    controller=uni.arc_control,
                    dynamic_params= uni.UnicycleParams(),
                    control_params=arc_params,
                    n_inputs=UnicycleControl.n_inputs,
                    plots=plot_manifest)

    # Run the simulation
    start_sim(sim=sim)

def run_differential_drive_arc_example() -> None:
    """Runs an example of a differential vehicle executing an arc

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
    params.tf = 5.
    sim = SimpleSim(params=params,
                    dynamics=diff.dynamics, # Differential drive
                    controller=diff.arc_control,
                    dynamic_params=diff.DiffDriveParams(L = 0.25, R=0.025),
                    control_params=arc_params,
                    n_inputs=UnicycleControl.n_inputs,
                    plots=plot_manifest)

    # Run the simulation
    start_sim(sim=sim)

def run_bicycle_arc_example() -> None:
    """Runs an example of a bicycle vehicle executing an arc

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
    params.tf = 5.
    sim = SimpleSim(params=params,
                    dynamics=bike.dynamics, # Bicycle
                    controller=bike.arc_control,
                    dynamic_params=bike.BicycleParams(L = 1.),
                    control_params=arc_params,
                    n_inputs=UnicycleControl.n_inputs,
                    plots=plot_manifest)

    # Run the simulation
    start_sim(sim=sim)

def simple_field_unicycle() -> None:
    """Runs an example of a go-to-goal vector field combined with an obstacle avoidance
    field with a unicycle dynamic model.
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
    start_sim(sim=sim)

def carrot_follow_unicycle() -> None:
    """Runs an example of a go-to-goal vector field combined with obstacle avoidance to show off the sensor measurements being performed. The optional ability to follow a path allows the vehicle to navigate around complex obstacles.
    """

    # Initialize the state and control
    vel_params = uni.UniVelVecParams(vd_field_max=5., k_wd= 5.)
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)

    # Create the vector field
    n_lines = 10 # Number of sensor lines
    vector_field = G2GAvoid(x_g=TwoDimArray(x=8., y=5.),
                            n_obs=n_lines,
                            v_max=vel_params.vd_field_max,
                            S=1.5,
                            R=1.,
                            sig=1.,
                            weight_g2g=1.,
                            weight_avoid=1.)

    # Create the obstacle world
    obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create the plan
    plan = create_path(start=TwoDimArray(x=state_initial.x, y=state_initial.y), end=vector_field.x_g, obstacle_world=obstacle_world, plan_type="visibility")
    if plan is not None:
        line = np.array([plan[0], plan[1]])
        carrot = LineCarrot(line=line, s_dev_max=5., s_carrot=2.)
    else:
        carrot = None

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 position_dot=False,
                                 position_triangle=True,
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
    params.sim_step = 0.01
    params.sim_update_period = 0.01
    params.tf = 5.
    sim = NavVectorFollower(params=params,
                            dynamics=uni.dynamics,
                            controller=uni.velocity_vector_field_control,
                            dynamic_params= uni.UnicycleParams(),
                            control_params=vel_params,
                            n_inputs=UnicycleControl.n_inputs,
                            plots=plot_manifest,
                            vector_field=vector_field,
                            world=obstacle_world,
                            sensor=RangeBearingSensor(n_lines=n_lines, max_dist=4.),
                            carrot=carrot
                         )

    # Run the simulation
    start_sim(sim=sim)

def dwa_scaled_plotting() -> None:
    """Runs a loop for plotting the scaled velocities for the dynamic window approach.
    """
    # Initialize the dwa search parameters
    ds = 0.1
    params = DwaParams(v_des=2.,
                       w_max=5.,
                       w_res=0.1,
                       ds=ds,
                       sf=2.,
                       s_eps=0.1,
                       k_v=2.,
                       sigma=2.,
                       classic=False,
                       v_res=0.25)
    obstacle_world = poly_world.generate_world_obstacles()

    # Create an initial state of the vehicle
    x0 = UnicycleState(x=3., y=2., psi=np.pi/4.)

    # Create a plot from the plot manifest
    plt_dist = 2.
    manifest = create_plot_manifest(initial_state=x0,
                                    world=obstacle_world,
                                    y_limits=(x0.y-plt_dist, x0.y+plt_dist),
                                    x_limits=(x0.x-plt_dist, x0.x+plt_dist),
                                    position_triangle=True,
                                    plot_occupancy_grid=True
                                    )
    for plot in manifest.state_plots:
        plot.plot(state=x0)
    for fig in manifest.figs:
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.show(block=False)

    # Loop through and calculate all of the resulting arcs
    for w in params.w_vals:
        # Calculate the time of collision
        cont_w = UnicycleControl(v=params.v_des, w=w)
        t_coll = dwa.evaluate_arc_collision(state=x0,
                                            params=params,
                                            control=cont_w,
                                            world=obstacle_world
                                            )

        # Scale the velocities
        scaled_vels = dwa.scale_velocities(control=cont_w, t_coll=t_coll, tf=params.tf)

        # Get the resulting arcs
        x_des, y_des = unicycle_solution_trajectory(init=x0, control=cont_w, ds=ds, tf=params.tf )
        x_act, y_act = unicycle_solution_trajectory(init=x0, control=scaled_vels, ds=ds, tf=params.tf)

        # Plot the arcs
        manifest.vehicle_axes.plot(x_des, y_des, 'b', linewidth=2)
        manifest.vehicle_axes.plot(x_act, y_act, 'k', linewidth=2)

    plt.show(block=True)

def run_dwa_unicycle() -> None:
    """Runs an example of the dynamic window approach with a unicycle dynamic model.
    """

    # Initialize the state and control
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)

    # Initialize the dwa search parameters
    ds = 0.05
    dwa_params = DwaParams(v_des=2.,
                           w_max=5.,
                           w_res=0.1,
                           ds=ds,
                           sf=2.,
                           s_eps=0.1,
                           k_v=2.,
                           sigma=2.,
                           classic=False,
                           v_res=0.25)

    # Create the obstacle world
    obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create an inflated grid from the world
    grid = og.generate_occupancy_from_polygon_world(
        world=obstacle_world,
        res=0.25,
        x_lim=(-2, 10),
        y_lim=(-2, 10))
    inf_grid = og.inflate_obstacles(grid=grid, inflation=0.25)

    # Create the plan to follow
    x_g = TwoDimArray(x=7., y=4.)
    plan = create_path(start=TwoDimArray(x=state_initial.x, y=state_initial.y),
                       end=x_g,
                       obstacle_world=obstacle_world, plan_type="visibility")
    if plan is None:
        raise ValueError("No plan was found")
    line = np.array([plan[0], plan[1]])
    carrot = LineCarrot(line=line, s_dev_max=5., s_carrot=2.)

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(
        initial_state=state_initial,
        y_limits=(-2, 10),
        x_limits=(-2, 10),
        position_dot=False,
        position_triangle=True,
        state_trajectory=True,
        time_series=True,
        vector_res=0.5,
        world=obstacle_world,
        plan=plan,
        line_carrot=carrot,
        grid=inf_grid,
        plot_occupancy_grid=True
        )

    # Create the simulation
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.1
    params.sim_step = 0.1
    params.sim_update_period = 0.1
    sim = DwaFollower(params=params,
                      dynamics= uni.dynamics,
                      controller=uni.arc_control,
                      dynamic_params= uni.UnicycleParams(),
                      dwa_params=dwa_params,
                      n_inputs=UnicycleControl.n_inputs,
                      plots=plot_manifest,
                      world=inf_grid,
                      carrot=carrot
                      )

    # Run the simulation
    start_sim(sim=sim)

if __name__ == "__main__":
    run_unicycle_arc_example()
    #run_differential_drive_arc_example()
    #run_bicycle_arc_example()
    #simple_field_unicycle()
    #carrot_follow_unicycle()
    #dwa_scaled_plotting()
    #run_dwa_unicycle()
