"""types.py: Defines several types that are used throughout the code

"""

import copy
from typing import Any, Generic, Optional, Protocol, TypeVar, cast

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
    n_inputs: int # The number of inputs (i.e., rows in input)

StateType = TypeVar("StateType", bound=State)
InputType = TypeVar("InputType", bound=Input)
ControlParamType = TypeVar("ControlParamType")

class Slice(Generic[StateType]):
    """ Contains a "slice" of data - the data produced / needed
        at a single time
    """
    def __init__(self, state: StateType, time: float = 0.) -> None:
        self.time = time # Simulation time corresponding to the state
        self.state: StateType = state # State
        self.input_vec: Optional[npt.NDArray[Any]] = None # Input applied at the stated time, None => not yet calculated

class Data(Generic[StateType]):
    """Stores the changing simulation information"""
    def __init__(self, current: Slice[StateType]) -> None:
        self.current = current # Stores the current slice of data to be read
        self.next = copy.deepcopy(current) # Stores the next data to be created
        self.state_traj: npt.NDArray[Any] # Each column corresponds to a trajectory data
        self.time_traj: npt.NDArray[Any] # vector Each element is the time for the state in question
        self.traj_index_latest: int = -1 # Index into the state and time trajectory of the latest data
        self.control_traj: npt.NDArray[Any] # Each column corresponds to a control input vector

    def get_state_vec(self,index: int) -> npt.NDArray[Any]:
        """Returns a state vector of valid values"""
        return self.state_traj[index, 0:self.traj_index_latest+1] # +1 as python is non-inclusive on second argument

    def get_time_vec(self) -> npt.NDArray[Any]:
        """Returns the valid time vector"""
        return self.time_traj[0:self.traj_index_latest+1]

    def get_control_vec(self, index: int) -> npt.NDArray[Any]:
        """Returns the control vector of valid values"""
        return self.control_traj[index, 0:self.traj_index_latest+1] # +1 as python is non-inclusive on the second argument

class TwoDArrayType(Protocol):
    """Defines a Two dimensional array with an x and y component"""
    IND_X: int # The index of the x-component
    IND_Y: int # The index of the y-component
    state: npt.NDArray[Any] # The 2-D array
    n_states: int # The number of states in state
    x: float # The value of the x component
    y: float # The value of the y component

class TwoDimArray:
    """Provides a representation of a two dimensional array
    """
    IND_X: int = 0 # The index of the x-component
    IND_Y: int = 1  # The index of the y-component
    n_states: int = 2 # The number of states in state

    def __init__(self, x: float = 0., y: float = 0., vec: Optional[npt.NDArray[Any]] = None) -> None:
        """Initializes the vector. If vec is defined then the x and y values are ignored"""
        # Extract values from vec to avoid numpy shape issues
        if vec is not None:
            x = vec.item(self.IND_X)
            y = vec.item(self.IND_Y)
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

class Dynamics(Protocol[StateType, InputType]): # type: ignore
    """Class taking the form of a state dynamics function call"""
    def __call__(self, state: StateType, control: InputType) -> StateType:
        """Dynamic function call ( xdot = f(x,u) )"""

class Control(Protocol[StateType, InputType, ControlParamType]): # type: ignore
    """Class taking the form of the control function"""
    def __call__(self, time: float, state: StateType, params: ControlParamType) -> InputType:
        """Control function call (u = g(t, x, P))"""

class VectorControl(Protocol[StateType, InputType, ControlParamType]): # type: ignore
    """Class taking the form of the control function"""
    def __call__(self, time: float, state: StateType, vec: TwoDimArray, params: ControlParamType) -> InputType:
        """Control function call (u = g(t, x, vec, P)) where vec is the desired vector to follow
        """

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
    position: npt.NDArray[Any] # The position vector of the vehicle
    n_states: int # The number of states in the state vector
    x: float # The x-position
    y: float # The y-position
    psi: float # The orientation

class UnicyleControlProtocol(Protocol):
    """Protocol for the input when calling the unicycle functions"""
    IND_V: int # The index of the translational velocity input
    IND_W: int # The index of the rotational velocity input
    input: npt.NDArray[Any] # The input vector
    n_inputs: int # The number of inputs (i.e., rows in input)
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

    @property
    def position(self) -> npt.NDArray[Any]:
        """Return the 2D position vector"""
        return np.array([[self.x], [self.y]])

    @position.setter
    def position(self, val: npt.NDArray[Any])->None:
        """Sets the 2D position vector"""
        self.x = val.item(self.IND_X)
        self.y = val.item(self.IND_Y)

UnicycleStateType = TypeVar("UnicycleStateType", bound=UnicycleState)

class UnicycleControl:
    """Stores the inputs required for the Unicycle dynamics (translational and rotation velocity)"""
    IND_V: int = 0# The index of the translational velocity input
    IND_W: int = 1# The index of the rotational velocity input
    n_inputs: int = 2 # The number of inputs (i.e., rows in input)

    def __init__(self, v: float = 0., w: float = 0., vec: Optional[npt.NDArray[Any]] = None) -> None:
        """Initializes the input vector from the passed in values. If vec is provided then v and w are ignored"""
        # Get the default values from the vector
        if vec is not None:
            v = vec.item(self.IND_V)
            w = vec.item(self.IND_W)

        # Create the input vector. Note that this avoids any issues with vec being a different shape
        # or modified elsewhere.
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

class VectorField(Protocol):
    """Defines the functions needed for a vector field class"""
    def calculate_vector(self, state: UnicyleStateProtocol, time: float) ->TwoDimArray:
        """Calculates a vector given the time and unicycle state"""
