"""dynamics_gen.py: Creating a generic dynamics class for inheritance
"""

from typing import Any, Callable, NewType, Protocol, TypeVar
from py_sim.tools.sim_types import ArcParams
from py_sim.Dynamics.unicycle import UnicycleInput, UnicycleState

from py_sim.tools.sim_types import State as StateIn
from py_sim.tools.sim_types import Input

State = TypeVar("State", bound=StateIn)
Control = TypeVar("Control", bound=Input)



class Dynamics(Protocol[State, Control]):
#class Dynamics(Protocol):
    @staticmethod
    def dynamics(state: State, control: Control) -> State:
        """Calculate the time derivative of the state"""

    @staticmethod
    def arc_control(time: float, state: State, params: ArcParams) -> Control:
        """Calculate control to follow an arc"""


class UniClass():
    @staticmethod
    def dynamics(state: UnicycleState, control: UnicycleInput) -> UnicycleState:
        return UnicycleState()

    @staticmethod
    def arc_control(time: float, state: UnicycleState, params: ArcParams) -> UnicycleInput:
        return UnicycleInput()

t: Dynamics[UnicycleState, UnicycleInput] = UniClass()
#t: Dynamics = UniClass()
