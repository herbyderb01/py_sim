

from py_sim.tools.sim_types import State as StateIn
from py_sim.tools.sim_types import Input as Input_type
from py_sim.Dynamics.unicycle import UnicycleState, UnicycleInput
import numpy as np
import numpy.typing as npt
from typing import Protocol, Any, Callable, TypeVar

State = TypeVar("State", bound=StateIn)
Control = TypeVar("Control", bound=Input_type)

#Dynamics = Callable[[UnicycleState, UnicycleInput], State]
Dynamics = Callable[[State, Control], State]

class Input(Protocol):
    """Protocol for the input when calling the unicycle functions"""
    IND_V: int # The index of the translational velocity input
    IND_W: int # The index of the rotational velocity input
    input: npt.NDArray[Any] # The input vector
    v: float # The value of the translational velocity
    w: float # The value of the rotational velocity

#def dynamics(state: State, control: Input_type) -> UnicycleState:
def dynamics(state: UnicycleState, control: Input_type) -> UnicycleState:
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
    # state_dot.x = control.v * np.cos(state.psi)
    # state_dot.y = control.v * np.sin(state.psi)
    # state_dot.psi = control.w
    return state_dot

v: Dynamics[UnicycleState, Input_type] = dynamics


## Now try inheritance
class StateSup:
    def __init__(self) -> None:
        self.state: npt.NDArray[Any]


class UniState(StateSup):
    def __init__(self) -> None:
        super().__init__()
        self.w: float

#dyn = Callable[[StateSup], StateSup]
dyn = Callable[[UniState], StateSup]

#def dyn_unic(state: UniState) -> UniState:
def dyn_unic(state: StateSup) -> UniState:
    var = UniState()
    return var

t: dyn = dyn_unic
