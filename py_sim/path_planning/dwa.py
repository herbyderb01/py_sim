""" dwa.py provides functions and classes for implementing a dynamic window controller
"""

import numpy as np
import numpy.typing as npt
from py_sim.worlds.polygon_world import PolygonWorld as World
from py_sim.tools.sim_types import TwoDArrayType, UnicycleStateProtocol, UnicycleControl
from typing import Protocol, Any
from py_sim.dynamics.unicycle import solution as unicycle_solution
from py_sim.tools.sim_types import Control, LocationStateType, InputType, DynamicsParamType, ArcParams, DwaParams
from typing import Generic

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
    scale = t_coll/tf # Scaling for collision
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
    # Initialize search values to occur at the current state
    dist = np.linalg.norm(state.position - goal.position)
    control = UnicycleControl(v=0., w=0.)

    # Loop through and find the control values that will result in the state closest to the end state
    for w in params.w_vals:
        # Calculate the time of collision
        cont_w = UnicycleControl(v=params.v_des, w=w)
        t_coll = evaluate_arc_collision(state=state,
                                        params=params,
                                        control=cont_w,
                                        world=world)

        # Scale the velocities
        scaled_vels = scale_velocities(control=cont_w, t_coll=t_coll, tf=params.tf)

        # Compare the resulting position with the previously best found
        state_scaled = unicycle_solution(init=state, control=scaled_vels, delta_t=params.tf)
        dist_scaled = np.linalg.norm(state_scaled.position-goal.position)
        if dist_scaled < dist:
            dist = dist_scaled
            control = scaled_vels

    return control
