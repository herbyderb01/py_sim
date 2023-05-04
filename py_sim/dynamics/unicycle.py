"""unicycle.py: Defines types, functions, and classes for the dynamics of a unicycle robot

Functions:

Classes:

"""
import numpy as np
from py_sim.tools.sim_types import (
    ArcParams,
    UnicycleControl,
    UnicycleState,
    UnicyleControlProtocol,
    UnicyleStateProtocol,
)


def dynamics(state: UnicyleStateProtocol, control: UnicyleControlProtocol) -> UnicycleState:
    """ Calculates the dynamics of the unicycle. Note that even though a "UnicycleState"
        is returned, that actually corresponds to the time derivative
            d/dt x = v cos(psi)
            d/dt y = v sin(spi)
            d/dt psi = w

        Controls:
            state: The current state of the vehicle
            control: The current input of the vehicle

        Outputs:
            The resulting state time derivative
    """
    state_dot = UnicycleState() # Time derivative of the state
    state_dot.x = control.v * np.cos(state.psi)
    state_dot.y = control.v * np.sin(state.psi)
    state_dot.psi = control.w
    return state_dot

def arc_control(time: float, state: UnicyleStateProtocol, params: ArcParams) -> UnicycleControl: # pylint: disable=unused-argument
    """ Implements the control for a circular arc
    """
    return UnicycleControl(v=params.v_d, w=params.w_d)
