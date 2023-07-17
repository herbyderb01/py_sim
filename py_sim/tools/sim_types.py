"""types.py: Defines several types that are used throughout the code

Attributes:
    StateType(TypeVar): A type bound by State
    InputType(TypeVar): A type bound by Input
    ControlParamType(TypeVar): A type placeholder for any control parameters
    UnicycleStateType(TypeVar): A type bound by UnicycleState
"""

import copy
from typing import Any, Generic, Optional, Protocol, TypeVar, cast, runtime_checkable

import numpy as np
import numpy.typing as npt


# Generic definitions for the state and input
class State(Protocol):
    """The basic form of a state

    Attributes:
        state(NDArray[Any]): The state vector of the vehicle
        n_states(int): The number of elements in the state
    """
    state: npt.NDArray[Any]
    n_states: int

class Input(Protocol):
    """The basic form of a control input

    Attributes:
        input(NDArray[Any]): A vector of inputs
        n_inputs(int): The number of inputs (i.e., rows in input)
    """
    input: npt.NDArray[Any]
    n_inputs: int

StateType = TypeVar("StateType", bound=State)
InputType = TypeVar("InputType", bound=Input)
ControlParamType = TypeVar("ControlParamType")

class Slice(Generic[StateType]):
    """ Contains a "slice" of data - the data produced / needed
        at a single time

    Attributes:
        time(float): Simulation time corresponding to the state
        state(StateType): State of the vehicle
        input_vec(Optional[npt.NDArray[Any]]): Input applied at the stated time, None => not yet calculated
    """
    def __init__(self, state: StateType, time: float = 0.) -> None:
        self.time = time # Simulation time corresponding to the state
        self.state: StateType = state # State
        self.input_vec: Optional[npt.NDArray[Any]] = None # Input applied at the stated time, None => not yet calculated

class Data(Generic[StateType]):
    """Stores the changing simulation information

    Attributes:
        current(Slice): Stores the current slice of data to be read
        next(Slice): Stores the next data to be created
        state_traj(NDArray[Any]): Each column corresponds to a trajectory data
        time_traj(NDArray[Any]): vector Each element is the time for the state in question
        traj_index_latest(int): Index into the state and time trajectory of the latest data
        control_traj(NDArray[Any]): Each column corresponds to a control input vector
        range_bearing_latest(RangeBearingMeasurements): Stores the latest data received for range-bearing measurements
    """
    def __init__(self, current: Slice[StateType]) -> None:
        self.current = current # Stores the current slice of data to be read
        self.next = copy.deepcopy(current) # Stores the next data to be created
        self.state_traj: npt.NDArray[Any] # Each column corresponds to a trajectory data
        self.time_traj: npt.NDArray[Any] # vector Each element is the time for the state in question
        self.traj_index_latest: int = -1 # Index into the state and time trajectory of the latest data
        self.control_traj: npt.NDArray[Any] # Each column corresponds to a control input vector
        self.range_bearing_latest = RangeBearingMeasurements() # Stores the latest data received for range-bearing measurements

    def get_state_vec(self,index: int) -> npt.NDArray[Any]:
        """Returns a vector of the valid values for a given state

        Args:
            index: The index of the state being requested

        Returns:
            NDArray[Any]: The array of state values from initial time to current time
        """
        return self.state_traj[index, 0:self.traj_index_latest+1] # +1 as python is non-inclusive on second argument

    def get_time_vec(self) -> npt.NDArray[Any]:
        """Returns the vector of valid time values from the initial time to the current time"""
        return self.time_traj[0:self.traj_index_latest+1]

    def get_control_vec(self, index: int) -> npt.NDArray[Any]:
        """Returns the control referenced by index over all valid time values

        Args:
            index: The index of the desired control input within the control vector

        Returns:
            NDArray[Any]: The requested control over time
        """
        return self.control_traj[index, 0:self.traj_index_latest+1] # +1 as python is non-inclusive on the second argument

class TwoDArrayType(Protocol):
    """Defines a Two dimensional array with an x and y component

    Attributes:
        IND_X(int): The index of the x-component
        IND_Y(int): The index of the y-component
        state(NDArray[Any]): The 2-D array
        n_states(int): The number of states in state
        x(float): The value of the x component
        y(float): The value of the y component
    """
    IND_X: int # The index of the x-component
    IND_Y: int # The index of the y-component
    state: npt.NDArray[Any] # The 2-D array
    position: npt.NDArray[Any] # The 2-D array, same as state
    n_states: int # The number of states in state
    x: float # The value of the x component
    y: float # The value of the y component

LocationStateType = TypeVar("LocationStateType", bound=TwoDArrayType)
class TwoDimArray:
    """Provides a representation of a two dimensional array

    Attributes:
        IND_X(int): The index of the x-component
        IND_Y(int): The index of the y-component
        state(NDArray[Any]): The 2-D array
        n_states(int): The number of states in state
        x(float): The value of the x component
        y(float): The value of the y component
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

    @property
    def position(self) -> npt.NDArray[Any]:
        """Returns the (x,y) position"""
        return self.state

    @position.setter
    def position(self, val: npt.NDArray[Any]) -> None:
        """Sets the position"""
        self.state.itemset(self.IND_X, val.item(self.IND_X))
        self.state.itemset(self.IND_Y, val.item(self.IND_Y))

class Dynamics(Protocol[StateType, InputType]): # type: ignore
    """Class taking the form of a state dynamics function call"""
    def __call__(self, state: StateType, control: InputType) -> StateType:
        """Dynamic function call ( xdot = f(x,u) )

        Args:
            state: The current state
            control: The current control input

        Returns:
            StateType: The time derivative of the state
        """

class Control(Protocol[StateType, InputType, ControlParamType]): # type: ignore
    """Class taking the form of the control function"""
    def __call__(self, time: float, state: StateType, params: ControlParamType) -> InputType:
        """Control function call (u = g(t, x, P))

        Args:
            time: The time for which the control is being calculated
            state: The state at the time the control is calculated
            params: The paramters for the control law

        Returns:
            InputType: The resulting control input
        """

class VectorControl(Protocol[StateType, InputType, ControlParamType]): # type: ignore
    """Class taking the form of the control function for a vector field"""
    def __call__(self, time: float, state: StateType, vec: TwoDimArray, params: ControlParamType) -> InputType:
        """Control function call (u = g(t, x, vec, P)) where vec is the desired vector to follow

        Args:
            time: The time for which the control is being calculated
            state: The state at the time the control is calculated
            vec: The vector to be followed
            params: The paramters for the control law

        Returns:
            InputType: The resulting control input
        """

class ArcParams():
    """Parameters required for defining an arc

    Args:
        v_d(float): Desired translational velocity
        w_d(float): Desired rotational velocity
    """
    def __init__(self, v_d: float = 0., w_d: float = 0.) -> None:
        self.v_d = v_d # Desired translational velocity
        self.w_d = w_d # Desired rotational velocity

@runtime_checkable
class UnicycleStateProtocol(Protocol):
    """Protocol for the state when calling the unicycle functions

    Attributes:
        IND_X(int): The index of the x-position
        IND_Y(int): The index of the y-position
        IND_PSI(int): The index of the orientation
        state(NDArray[Any]): The state vector of the vehicle
        position(NDArray[Any]): The position vector of the vehicle
        n_states(int): The number of states in the state vector
        x(float): The x-position
        y(float): The y-position
        psi(float): The orientation

    """
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
    """Protocol for the input when calling the unicycle functions

    Attributes:
        IND_V(int): The index of the translational velocity input
        IND_W(int): The index of the rotational velocity input
        input(NDArray[Any]): The input vector
        n_inputs(int): The number of inputs (i.e., rows in input)
        v(float): The value of the translational velocity
        w(float): The value of the rotational velocity
    """
    IND_V: int # The index of the translational velocity input
    IND_W: int # The index of the rotational velocity input
    input: npt.NDArray[Any] # The input vector
    n_inputs: int # The number of inputs (i.e., rows in input)
    v: float # The value of the translational velocity
    w: float # The value of the rotational velocity

class UnicycleState:
    """Provides a representation of the vehicle whose state consists of
       an (x,y) position and an orientation (psi)

    Attributes:
        IND_X(int): The index of the x-position
        IND_Y(int): The index of the y-position
        IND_PSI(int): The index of the orientation
        state(NDArray[Any]): The state vector of the vehicle
        position(NDArray[Any]): The position vector of the vehicle
        n_states(int): The number of states in the state vector
        x(float): The x-position
        y(float): The y-position
        psi(float): The orientation
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
    """Stores the inputs required for the Unicycle dynamics (translational and rotation velocity)

    Attributes:
        IND_V(int): The index of the translational velocity input
        IND_W(int): The index of the rotational velocity input
        input(NDArray[Any]): The input vector
        n_inputs(int): The number of inputs (i.e., rows in input)
        v(float): The value of the translational velocity
        w(float): The value of the rotational velocity
    """
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
    def calculate_vector(self, state: TwoDArrayType, time: float) ->TwoDimArray:
        """Calculates a vector given the time and unicycle state

        Args:
            state: Current position and orientation
            time: time in question (seconds)

        Returns:
            TwoDimArray: The resulting vector
        """

class RangeBearingMeasurements:
    """Storage for range and bearing measurements

    Attributes:
        range(list[float]): Range to a number of measurements
        bearing(list[float]): Bearing to the measurements
        location(list[TwoDimArray]): Location of each of the measurements
    """
    def __init__(self) -> None:
        """Creates empty vectors of range and bearing measurements"""
        self.range: list[float] = [] # Range to a number of measurements
        self.bearing: list[float] = [] # Bearing to the measurements
        self.location: list[TwoDimArray] = [] # Location of each of the measurements

class StateSpace:
    """Defines the rectangular limits of a state space

    Attributes:
        x_lim(tuple[float, float]): Lower and upper limit for the x value
        y_lim(tuple[float, float]): Lower and upper limit for the y value
    """
    def __init__(self, x_lim: tuple[float, float], y_lim: tuple[float, float]) -> None:
        """ Initializes the state space limits. Note that the limits must be increasing.

            Args:
                x_lim: Lower and upper limit for the x value
                y_lim: lower and upper limit for the y value
        """
        # Store the data
        self.x_lim = x_lim
        self.y_lim = y_lim

        # Check the limits
        if x_lim[1] < x_lim[0] or y_lim[1] < y_lim[0]:
            raise ValueError("Limits must be increasing")

    def contains(self, state: TwoDimArray) -> bool:
        """ Evaluates if the state is in the state space. Returns true if it is

        Args:
            state: State to be evaluated

        Returns:
            bool: True if the state is in the state space, false otherwise
        """
        return bool(state.x >= self.x_lim[0] and state.x <= self.x_lim[1] and \
                    state.y >= self.y_lim[0] and state.y <= self.y_lim[1])

    def furthest_point(self, x: TwoDimArray) -> TwoDimArray:
        """Returns the furthest point in the state space furthest from x

        Args:
            x: An point to evaluate

        Returns:
            TwoDimArray: The furthest point from x
        """
        # Evaluate bottom left corner
        x_out = TwoDimArray(x=self.x_lim[0], y=self.y_lim[0])
        dist = np.linalg.norm(x_out.state-x.state)

        # Evaluate top left corner
        x_tl = TwoDimArray(x=self.x_lim[0], y=self.y_lim[1])
        dist_tl = np.linalg.norm(x_tl.state-x.state)
        if dist_tl > dist:
            dist = dist_tl
            x_out = x_tl

        # Evaluate top right corner
        x_tr = TwoDimArray(x=self.x_lim[1], y=self.y_lim[1])
        dist_tr = np.linalg.norm(x_tr.state-x.state)
        if dist_tr > dist:
            dist = dist_tr
            x_out = x_tr

        # Evaluate bottom right corner
        x_br = TwoDimArray(x=self.x_lim[1], y=self.y_lim[0])
        dist_br = np.linalg.norm(x_br.state-x.state)
        if dist_br > dist:
            dist = dist_br
            x_out = x_br

        return x_out

class EllipseParameters:
    """ Defines the parameters necessary for defining an ellipse

    Attributes:
        a: Major axis radius
        b: Minor axis radius
        center: Center point of the ellipse
        alpha: orientation from x-axis of ellipse
        c_a: the cosine of alpha
        s_a: the sine of alpha
    """
    def __init__(self, a: float = 0., b: float = 0., center: TwoDimArray = TwoDimArray(), alpha: float = 0.) -> None:
        """Store the parameters"""
        self.a: float = a
        self.b: float = b
        self.center: TwoDimArray = center
        self._alpha: float = alpha
        self._c_a: float = np.cos(alpha)
        self._s_a: float = np.sin(alpha)

    @property
    def alpha(self) -> float:
        """Return the alpha component"""
        return self._alpha
    @alpha.setter
    def alpha(self, val: float) -> None:
        """Sets alpha and the trig function values"""
        self._c_a = np.cos(val)
        self._s_a = np.sin(val)
        self._alpha = val

    @property
    def c_a(self) -> float:
        """Returns the cosine of alpha"""
        return self._c_a

    @property
    def s_a(self) -> float:
        """Returns the sine of alpha"""
        return self._s_a
