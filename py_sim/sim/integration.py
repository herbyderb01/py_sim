"""integration.py defines functions for numerical integration
"""
from typing import Any, cast

import numpy.typing as npt
from py_sim.tools.sim_types import Dynamics, InputType, StateType


def euler_update(dynamics: Dynamics[StateType, InputType], initial: StateType, control: InputType, dt: float) -> npt.NDArray[Any]:
    """Performs an eulers update to simulate forward one step on the dynamics

    Args:
        dynamics: a function handle for calculating the time derivative of the state
        initial: the starting state of the vehicle
        control: the control input to be applied
        dt: the time step of the update

    Returns:
        NDArray[Any]: The resulting time derivative of the state
    """
    result = initial.state + dt*( dynamics(initial, control).state )
    return cast(npt.NDArray[Any], result)
