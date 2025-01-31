""" dwa.py provides functions and classes for implementing a dynamic window controller
"""

from typing import Any, Protocol

import numpy as np
import numpy.typing as npt
from py_sim.dynamics.unicycle import solution as unicycle_solution
from py_sim.tools.sim_types import (
    DwaParams,
    TwoDArrayType,
    UnicycleControl,
    UnicycleStateProtocol,
)


class World(Protocol):
    """Defines a world in which a point can be queried for obstacle collision
    """
    def inside_obstacle(self, point: npt.NDArray[Any]) -> bool:
        """Returns true if the given point is inside any of the polygons defining the world

        Args:
            Point: 2x1 point to be evaluated

        Returns:
            bool: True if the point is in an obstacle
        """

def evaluate_arc_collision(state: UnicycleStateProtocol,
                           params: DwaParams,
                           control: UnicycleControl,
                           world: World
                           ) -> float:
    """Finds the largest value of time for which a collision will not occur along the arc

    Args:
        state: The starting state of the unicycle
        params: The parameters of the DWA
        control: The control being evaluated
        world: The world through which the vehicle is navigating

    Returns:
        float: The maximum time value without a collision
    """
    # Loop through the time values and find a collision time
    for t in params.t_vals:

        # Evaluate state at time t for collision
        state_t = unicycle_solution(init=state, control=control, delta_t=t)
        if world.inside_obstacle(point=state_t.position):
            return t - params.t_eps # Return time of collision - epsilon

    return params.tf

def scale_velocities(control: UnicycleControl, t_coll: float, tf: float) -> UnicycleControl:
    """Scales the velocities so that the end of the horizon (tf) will occur before the collision

    Args:
        control: The desired control values
        t_coll: The time of collision
        tf: The desired final time

    Returns:
        UnicycleControl: The control values that will result in no collision before the final time
    """
    scale = t_coll/tf
    return UnicycleControl(v=control.v*scale, w=control.w*scale)

def compute_desired_velocities(state: UnicycleStateProtocol,
                               params: DwaParams,
                               goal: TwoDArrayType,
                               world: World) -> UnicycleControl:
    """Computes the desired velocities for the vehicle. This is effectively a single run of the DWA algorithm.

    Note that this is a simplification from typical DWA. In traditional DWA, a dynamic window around
    the current velocities is assumed. This function tries out all velocities within a window.
    An additional difference is that a line search is made over the rotational velocities and
    the resulting arc is scaled based on obstacles.

    The velocity pair that results in the closest proximity to the goal is chosen.

    Args:
        state: The current state of the system
        params: Parameters used to calculate the control
        goal: The goal position of the control
        world: The world in which the vehicle is operating, used to query for obstacle avoidance
    """
    # Calculate the desired velocity to reach the goal at the end of the horizon
    dist_max = np.linalg.norm(goal.position - state.position)
    vel_max = dist_max/params.tf
    vel_des = np.minimum(params.v_des, vel_max)

    # Initialize search values to occur at the current state
    cost_min = np.inf  # Lowest cost seen thus far
    control = UnicycleControl(v=0., w=0.)

    # Loop through and find the control values that will result in the state closest to the end state
    for w in params.w_vals:
        # Calculate the time of collision
        cont_w = UnicycleControl(v=vel_des, w=w)
        t_coll = evaluate_arc_collision(state=state,
                                        params=params,
                                        control=cont_w,
                                        world=world)


        # Scale the velocities
        scaled_vels = scale_velocities(control=cont_w, t_coll=t_coll, tf=params.tf)

        # Calculate the scaled state and distance to the goal
        state_scaled = unicycle_solution(init=state, control=scaled_vels, delta_t=params.tf)
        dist_scaled = np.linalg.norm(state_scaled.position-goal.position)/dist_max

        # Calculate cost component due to velocity
        cost_vel = params.k_v * (1-np.exp(-dist_max**2/params.sigma**2))*(params.v_des - scaled_vels.v)**2

        # Choose the minimum cost
        cost = dist_scaled + cost_vel
        if cost < cost_min:
            cost_min = cost
            control = scaled_vels

    return control

def compute_desired_velocities_classic(
        state: UnicycleStateProtocol,
        params: DwaParams,
        goal: TwoDArrayType,
        world: World
) -> UnicycleControl:
    """ Computes the desired velocities using the classic DWA algorithm where a grid of
    velocities is evaluated for the best control.

    Args:
        state: The current state of the system
        params: Parameters used to calculate the control
        goal: The goal position of the control
        world: The world in which the vehicle is operating, used to query for obstacle avoidance
    Returns:
        UnicycleControl: The control values that will result in the state closest to the goal
    """
    # Initialize search values to occur at the current state
    dist_max = np.linalg.norm(goal.position - state.position)
    vel_max = dist_max/params.tf
    vel_des = np.minimum(params.v_des, vel_max)
    cost_min = np.inf  # Lowest cost seen thus far
    control = UnicycleControl(v=0., w=0.)

    # Loop through and find the control values that will result in the lowest cost
    v_vals = np.arange(start=-vel_des, stop=vel_des+params.v_res, step=params.v_res)
    for v in v_vals:
        for w in params.w_vals:
            # Calculate the time of collision
            cont_vw = UnicycleControl(v=v, w=w)
            t_coll = evaluate_arc_collision(state=state,
                                            params=params,
                                            control=cont_vw,
                                            world=world)
            if t_coll < params.tf:
                continue

            # Compare the resulting position with the previously best found
            state_new = unicycle_solution(init=state, control=cont_vw, delta_t=params.tf)
            dist_scaled = np.linalg.norm(state_new.position-goal.position) / dist_max

            # Calculate cost component due to velocity
            cost_vel = params.k_v * np.exp(-dist_max**2/params.sigma**2)*(params.v_des - v)**2

            # Choose the minimum cost
            cost = dist_scaled + cost_vel
            if cost < cost_min:
                cost_min = cost
                control = cont_vw

    return control
