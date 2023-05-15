"""types.py: Defines several types that are used throughout the code

"""

from typing import Any, Protocol, TypeVar, cast

import numpy as np
import numpy.typing as npt


# Generic definitions for the state and input
class State(Protocol):
    """The basic form of a state"""
    state: npt.NDArray[Any] # The state vector of the vehicle
    n_states: int # The number of elements in the state

class Input(Protocol):
    """The basic form of a control input"""
    input: npt.NDArray[Any] # A vector of inputs


class TwoDArrayType(Protocol):
    """Defines a Two dimensional array with an x and y component"""
    IND_X: int # The index of the x-component
    IND_Y: int # The index of the y-component
    state: npt.NDArray[Any] # The 2-D array
    x: float # The value of the x component
    y: float # The value of the y component

class TwoDimArray:
    """Provides a representation of a two dimensional array
    """
    IND_X: int = 0 # The index of the x-component
    IND_Y: int = 1  # The index of the y-component

    def __init__(self, x: float = 0., y: float = 0.) -> None:
        self.state: npt.NDArray[Any] = np.array([[x], [y]])  # The 2-D vector

    @property
    def x(self) -> float:
        """Return the x-component value"""
        return float(self.state.item(self.IND_X))

    @x.setter
    def x(self, val: float) -> None:
        """Store the x-component value"""
        self.state[self.IND_X,0] = val

    @property
    def y(self) -> float:
        """Return the y-component value"""
        return float(self.state.item(self.IND_Y))

    @y.setter
    def y(self, val: float) -> None:
        """Store the y-component value"""
        self.state[self.IND_Y,0] = val

StateType = TypeVar("StateType", bound=State)
InputType = TypeVar("InputType", bound=Input)
ControlParamType = TypeVar("ControlParamType")

class Dynamics(Protocol[StateType, InputType]): # type: ignore
    """Class taking the form of a state dynamics function call"""
    def __call__(self, state: StateType, control: InputType) -> StateType:
        """Dynamic function call ( xdot = f(x,u) )"""

class Control(Protocol[StateType, InputType, ControlParamType]): # type: ignore
    """Class taking the form of the control function"""
    def __call__(self, time: float, state: StateType, params: ControlParamType) -> InputType:
        """Control function call (u = g(t, x, P))"""

class ArcParams():
    """Parameters required for defining an arc"""
    def __init__(self, v_d: float = 0., w_d: float = 0.) -> None:
        self.v_d = v_d # Desired translational velocity
        self.w_d = w_d # Desired rotational velocity

class UnicyleStateProtocol(Protocol):
    """Protocol for the state when calling the unicycle functions"""
    IND_X: int  # The index of the x-position
    IND_Y: int  # The index of the y-position
    IND_PSI: int  # The index of the orientation
    state: npt.NDArray[Any] # The state vector of the vehicle
    n_states: int # The number of states in the state vector
    x: float # The x-position
    y: float # The y-position
    psi: float # The orientation

class UnicyleControlProtocol(Protocol):
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
    n_states: int = 3 # The number of states in the state vector

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

class UnicycleControl:
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
