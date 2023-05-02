"""unicycle.py: Defines types, functions, and classes for the dynamics of a unicycle robot

Functions:

Classes:

"""
from typing import Any, Protocol, cast

import numpy as np
import numpy.typing as npt
from py_sim.tools.sim_types import ArcParams


class State(Protocol):
    """Protocol for the state when calling the unicycle functions"""
    IND_X: int  # The index of the x-position
    IND_Y: int  # The index of the y-position
    IND_PSI: int  # The index of the orientation
    state: npt.NDArray[Any] # The state vector of the vehicle
    x: float # The x-position
    y: float # The y-position
    psi: float # The orientation

class Input(Protocol):
    """Protocol for the input when calling the unicycle functions"""
    IND_V: int # The index of the translational velocity input
    IND_W: int # The index of the rotational velocity input
    input: npt.NDArray[Any] # The input vector
    v: float # The value of the translational velocity
    w: float # The value of the rotational velocity

class UnicycleState:
    """Provides a representation of the vehicle whose state consists of
       an (x,y) position and an orientation (psi)
    """
    IND_X: int = 0 # The index of the x-position
    IND_Y: int = 1  # The index of the y-position
    IND_PSI: int = 2  # The index of the orientation

    def __init__(self, x: float = 0., y: float = 0., psi: float = 0.) -> None:
        self.state: npt.NDArray[Any] = np.array([[x], [y], [psi]])  # The state of the vehicle

    @property
    def x(self) -> float:
        """Return the x-position value"""
        return cast(float, self.state.item(self.IND_X))

    @x.setter
    def x(self, val: float) -> None:
        """Store the x-position value"""
        self.state[self.IND_X,0] = val

    @property
    def y(self) -> float:
        """Return the y-position value"""
        return cast(float, self.state.item(self.IND_Y))

    @y.setter
    def y(self, val: float) -> None:
        """Store the y-position value"""
        self.state[self.IND_Y,0] = val

    @property
    def psi(self) -> float:
        """Return the orientation value"""
        return cast(float, self.state.item(self.IND_PSI))

    @psi.setter
    def psi(self, val: float) -> None:
        """Store the y-position value"""
        self.state[self.IND_PSI,0] = val

class UnicycleInput:
    """Stores the inputs required for the Unicycle dynamics (translational and rotation velocity)"""
    IND_V: int = 0# The index of the translational velocity input
    IND_W: int = 1# The index of the rotational velocity input

    def __init__(self, v: float = 0., w: float = 0.) -> None:
        """Initializes the input vector from the passed in values"""
        self.input = np.array([[v],[w]])

    @property
    def v(self) -> float:
        """Return the translational velocity"""
        return cast(float, self.input.item(self.IND_V))

    @v.setter
    def v(self, val: float) -> None:
        """Store the translational velocity"""
        self.input[self.IND_V,0] = val

    @property
    def w(self) -> float:
        """Return the rotational velocity value"""
        return cast(float, self.input.item(self.IND_W))

    @w.setter
    def w(self, val: float) -> None:
        """Store the rotational velocity value"""
        self.input[self.IND_W,0] = val

def dynamics(state: UnicycleState, control: Input) -> UnicycleState:
    """ Calculates the dynamics of the unicycle. Note that even though a "UnicycleState"
        is returned, that actually corresponds to the time derivative
            d/dt x = v cos(psi)
            d/dt y = v sin(spi)
            d/dt psi = w

        Inputs:
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

def arc_control(time: float, state: State, params: ArcParams) -> UnicycleInput: # pylint: disable=unused-argument
    """ Implements the control for a circular arc
    """
    return UnicycleInput(v=params.v_d, w=params.w_d)
