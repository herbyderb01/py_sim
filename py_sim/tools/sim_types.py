"""types.py: Defines several types that are used throughout the code

"""

from typing import Any, Callable, NewType, Protocol

import numpy as np
import numpy.typing as npt


# Generic definitions for the state and input
class State(Protocol):
    """The basic form of a state"""
    state: npt.NDArray[Any] # The state vector of the vehicle

class Input(Protocol):
    """The basic form of a control input"""
    input: npt.NDArray[Any] # A vector of inputs

# Dynamics function type
#   Inputs:
#       State: The current state of the vehicle
#       Input: The control input
#   Outputs:
#       State: The time derivative of the state being output
#Dynamics = NewType("Dynamics", Callable[[State, Input], State])
Dynamics = Callable[[State, Input], State]
# class Dynamics(Protocol):
#     """Class taking the form of a state dynamics function call"""
#     def __call__(self, state: State, control: Input) -> State:
#         """Dynamic function call ( xdot = f(x,u) )"""

# Dynamics update function type
#   Inputs:
#       dynamics: a function handle for calculating the time derivative of the state
#       initial: the starting state of the vehicle
#       input: the control input to be applied
#       dt: the time step of the update
#   Outputs:
#       The resulting time derivative of the state
# class DynamicsUpdate(Protocol):
#     """Class taking the form of the dynamic update call"""
#     def __call__(self, dynamics: Dynamics, initial: State, control: Input, dt: float) -> npt.NDArray[Any]:
#         """Calling function for the dynamic update"""
DynamicsUpdate = Callable[[Dynamics, State, Input, float], npt.NDArray[Any]]


class TwoDimVector:
    """Provides a representation of a two dimensional vector
    """
    IND_X: int = 0 # The index of the x-component
    IND_Y: int = 1  # The index of the y-component

    def __init__(self, x: float = 0., y: float = 0.) -> None:
        self.vec: npt.NDArray[Any] = np.array([[x], [y]])  # The 2-D vector

    @property
    def x(self) -> float:
        """Return the x-component value"""
        return float(self.vec.item(self.IND_X))

    @x.setter
    def x(self, val: float) -> None:
        """Store the x-component value"""
        self.vec[self.IND_X,0] = val

    @property
    def y(self) -> float:
        """Return the y-component value"""
        return float(self.vec.item(self.IND_Y))

    @y.setter
    def y(self, val: float) -> None:
        """Store the y-component value"""
        self.vec[self.IND_Y,0] = val

class ArcParams():
    """Parameters required for defining an arc"""
    def __init__(self, v_d: float = 0., w_d: float = 0.) -> None:
        self.v_d = v_d # Desired translational velocity
        self.w_d = w_d # Desired rotational velocity
