"""dwa_test.py: Provides a test for the dynamic window approach
"""

import time
from typing import Generic

import matplotlib.pyplot as plt
import py_sim.path_planning.forward_grid_search as search
import py_sim.worlds.polygon_world as poly_world
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.plotting.plotting import PlotManifest
from py_sim.sensors.occupancy_grid import generate_occupancy_from_polygon_world
from py_sim.sim.generic_sim import SimParameters, SingleAgentSim
from py_sim.tools.sim_types import (
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
    UnicycleStateType,
)
import py_sim.path_planning.dwa as dwa
from py_sim.dynamics.unicycle import solution as unicycle_solution

def get_arc(init: UnicycleStateType,
            vel: UnicycleControl,
            ds: float,
            tf: float) -> tuple[list[float], list[float]]:
    """Returns the (x,y) vector of a defined arc

    Args:
        init: initial state
        vel: the velocities being executed
        ds: The resolution in meters of the desired state spacing
        tf: The final time value of execution

    Returns:
        tuple[list[float], list[float]]: the x_vec and y_vec of the arc
    """
    # Initialize the outputs
    x_vec: list[float] = [init.x]
    y_vec: list[float] = [init.y]

    # Calculate the resolution of the time evaluations
    dt = ds/vel.v

    # Evaluate the time
    t = dt
    while t <= tf:
        # Calculate and store the position
        soln = unicycle_solution(init=init, control=vel, delta_t=t)
        x_vec.append(soln.x)
        y_vec.append(soln.y)

        # Update the time
        t += dt

    return (x_vec, y_vec)

def plot_arcs():
    """Plots example arcs produced by a dwa search"""

    # Initialize the dwa search parameters
    ds = 0.1
    params = dwa.DwaParams(v_des=2.,
                           w_max=1.,
                           w_res=0.1,
                           ds=ds,
                           sf=2.,
                           s_eps=0.1)
    obstacle_world = poly_world.generate_world_obstacles()
    # obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create an initial state and a goal state
    x0 = UnicycleState(x=1., y=1., psi=0.)
    xg = TwoDimArray(x=2, y=2.)

    # Calcuate the desired velocities
    vel_des = dwa.compute_desired_velocities(state=x0,
                                             params=params,
                                             goal=xg,
                                             world=obstacle_world)

    # Loop through and calculate all of the resulting arcs
    for w in params.w_vals:
        # Calculate the time of collision
        cont_w = UnicycleControl(v=params.v_des, w=w)
        t_coll = dwa.evaluate_arc_collision(state=x0,
                                            params=params,
                                            control=cont_w,
                                            world=obstacle_world)

        # Scale the velocities
        scaled_vels = dwa.scale_velocities(control=cont_w, t_coll=t_coll, tf=params.tf)

        # Get the resulting arcs
        x_des, y_des = get_arc(init=x0, vel=cont_w, ds=ds, tf=params.tf )
        x_act, y_act = get_arc(init=x0, vel=scaled_vels, ds=ds, tf=params.tf)