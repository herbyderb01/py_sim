""" dwa.py provides functions and classes for implementing a dynamic window controller
"""

import numpy as np
import numpy.typing as npt
from py_sim.worlds.polygon_world import PolygonWorld as World
from py_sim.tools.sim_types import TwoDArrayType, UnicycleStateProtocol, UnicycleControl
from typing import Protocol, Any

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

class DwaParams():
    """Parameters used for following a carrot point using the dynamic window approach

    Attributes:
        v_des(float): The desired translational velocity
        w_vals(list(float)): The desired rotational velocities to search
        _w_vals(list(float)): The desired rotational velocities to search (internal)
        _w_max(float): The maximum angular velocity to search
        _w_res(float): The resolution of the arc search. Arc searched
                      from -w_max:w_res:w_max
    """
    def __init__(self, v_des: float, w_max: float, w_res: float) -> None:
        """ Initializes the parameters of the DWA search

        Args:
            v_des: The desired translational velocity
            w_max: The maximum angular velocity to search
            w_res(float): The resolution of the arc search. Arc searched
                      from -w_max:w_res:w_max
        """
        # Check the inputs
        if v_des <= 0. or w_max <= 0. or w_res <= 0.:
            raise ValueError("DWA parameters must all be positive")

        # Store the inputs
        self.v_des = v_des
        self._w_max = w_max
        self._w_res = w_res

        # Create the range of rotational velocity values over which to search
        self._w_vals: list[float] = []
        for w in np.arange(start=-w_max, stop=w_max, step=w_res).tolist():
            self._w_vals.append(w)
        self._w_vals.append(w_max) # arange is not inclusive on the stop

    @property
    def w_vals(self) -> list[float]:
        """Returns the rotational velocity list"""
        return self._w_vals

def compute_desired_velocities(state: UnicycleStateProtocol,
                               params: DwaParams,
                               goal: TwoDArrayType,
                               world: World) -> UnicycleControl:
    """Computes the desired velocities for the vehicle.

    Note that this is a simplification from typical DWA where a dynamic window around the current velociteis is assumed.

    Args:
        state: The current state of the system
        params: Parameters used to calculate the control
        goal: The goal position of the control
        world: The world in which the vehicle is operating, used to query for obstacle avoidance
    """
    