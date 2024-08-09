"""dwa_test.py: Provides a test for the dynamic window approach
"""

import matplotlib.pyplot as plt
import numpy as np
import py_sim.dynamics.unicycle as uni
import py_sim.sensors.occupancy_grid as og
import py_sim.worlds.polygon_world as poly_world
from matplotlib.axes import Axes
from py_sim.dynamics.unicycle import solution_trajectory as unicycle_solution_trajectory
from py_sim.path_planning import dwa
from py_sim.path_planning.path_generation import create_path
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.sim.generic_sim import SimParameters, start_sim
from py_sim.sim.sim_modes import DwaFollower
from py_sim.tools.projections import LineCarrot
from py_sim.tools.sim_types import (
    DwaParams,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
)


def plot_arcs() -> None:
    """Plots example arcs produced by a dwa search"""

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
    # obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create an initial state and a goal state
    #x0 = UnicycleState(x=3., y=2., psi=np.pi/4.)
    x0 = UnicycleState(x=5.5005, y=0.33756, psi=-0.411)
    #xg = TwoDimArray(x=5, y=5.)
    xg = TwoDimArray(x=6.44, y=2.326)

    # Create an inflated grid from the world
    grid = og.generate_occupancy_from_polygon_world(
        world=obstacle_world,
        res=0.25,
        x_lim=(-5, 25),
        y_lim=(-5, 10))
    inf_grid = og.inflate_obstacles(grid=grid, inflation=0.25)

    # Create a plot from the plot manifest
    plt_dist = 5.
    manifest = create_plot_manifest(initial_state=x0,
                                    world=obstacle_world,
                                    y_limits=(x0.y-plt_dist, x0.y+plt_dist),
                                    x_limits=(x0.x-plt_dist, x0.x+plt_dist),
                                    grid=inf_grid,
                                    position_triangle=True,
                                    plot_occupancy_grid=True
                                    )
    for plot in manifest.state_plots:
        plot.plot(state=x0)
    for fig in manifest.figs:
        fig.canvas.draw()
        fig.canvas.flush_events()
    plt.show(block=False)

    # Plot the goal location
    v_plot: Axes = manifest.vehicle_axes
    v_plot.plot(xg.x, xg.y, 'go', linewidth=8)


    # Calcuate the desired velocities
    vel_des = dwa.compute_desired_velocities(state=x0,
                                             params=params,
                                             goal=xg,
                                             #world=obstacle_world)
                                             world=inf_grid)

    # Loop through and calculate all of the resulting arcs
    for w in params.w_vals:
        # Calculate the time of collision
        cont_w = UnicycleControl(v=params.v_des, w=w)
        t_coll = dwa.evaluate_arc_collision(state=x0,
                                            params=params,
                                            control=cont_w,
                                            #world=obstacle_world
                                            world=inf_grid)

        # Scale the velocities
        scaled_vels = dwa.scale_velocities(control=cont_w, t_coll=t_coll, tf=params.tf)

        # Get the resulting arcs
        x_des, y_des = unicycle_solution_trajectory(init=x0, control=cont_w, ds=ds, tf=params.tf )
        x_act, y_act = unicycle_solution_trajectory(init=x0, control=scaled_vels, ds=ds, tf=params.tf)

        # Plot the arcs
        v_plot.plot(x_des, y_des, 'b', linewidth=2)
        v_plot.plot(x_act, y_act, 'k', linewidth=2)

    # Plot the goal velocities
    x_g, y_g = unicycle_solution_trajectory(init=x0, control=vel_des, ds=ds, tf=params.tf)
    v_plot.plot(x_g, y_g, 'g', linewidth=3)

    plt.show(block=True)

def run_unicycle_dwa_example() -> None:
    """Runs an example of a DWA controller used with a unicycle
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
    obstacle_world = poly_world.generate_world_obstacles()
    #obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create an inflated grid from the world
    grid = og.generate_occupancy_from_polygon_world(
        world=obstacle_world,
        res=0.25,
        x_lim=(-5, 25),
        y_lim=(-5, 10))
    inf_grid = og.inflate_obstacles(grid=grid, inflation=0.25)

    # Create the plan to follow
    x_g = TwoDimArray(x=7., y=4.)
    #x_g = TwoDimArray(x=13., y=5.)
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
        y_limits=(-5, 10),
        x_limits=(-5, 25),
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
                      #world=obstacle_world,
                      #world=grid,
                      world=inf_grid,
                      carrot=carrot
                      )

    # Run the simulation
    start_sim(sim=sim)

def main() -> None:
    """Runs the dynamic window approach test"""
    # Simple script to visualize the arcs produced by the DWA search
    #plot_arcs()

    # Run the unicycle DWA example
    run_unicycle_dwa_example()

if __name__ == "__main__":
    main()
