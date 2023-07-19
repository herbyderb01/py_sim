"""single_integrator.py: Defines types, functions, and classes for controlling a point with single integrator dynamics (i.e., direct control of the velocity)
"""

from typing import Any

import numpy as np
import numpy.typing as npt
from py_sim.tools.sim_types import TwoDArrayType, TwoDimArray


class PointInput:
    """Defines the inputs for a single integrator system

    Attributes:
        input(NDArray[Any]): A vector of inputs
        n_inputs(int): The number of inputs (i.e., rows in input)
        vec(TwoDimArray): The vector storing the input (input reads from vec)
    """
    n_inputs: int = 2 # The number of inputs

    def __init__(self, vec: TwoDimArray) -> None:
        self.vec = vec

    @property
    def input(self) -> npt.NDArray[Any]:
        """Returns the control input"""
        return self.vec.state

    @input.setter
    def input(self, val: npt.NDArray[Any]) -> None:
        """Sets the input value"""
        self.vec.state = val


def dynamics(state: TwoDArrayType, control: PointInput) -> TwoDimArray: # pylint: disable=unused-argument
    """ Calculates single integrator dynamics.

    The returned time derivative corresponds to
        d/dt x
        d/dt y

    Args:
        state: The current position of the vehicle
        control: The current input (velocities)

    Returns:
        TwoDimArray: The resulting state time derivative (actually the input)
    """
    return control.vec

###########################  Basic controllers ##################################
#### Constant controller #####
class ConstantInputParams:
    """Parameters required for defining a constant velocity

    Attributes:
        v_d(TwoDimArray): Desired constant input
    """
    def __init__(self, v_d: TwoDimArray) -> None:
        self.v_d = v_d

def const_control(time: float, state: TwoDArrayType, params: ConstantInputParams) -> PointInput: # pylint: disable=unused-argument
    """Implements a constant control for the single integrator dynamics

    Args:
        time: clock time (not used)
        state: vehicle state (not used)
        params: Defines the desired velocity vector
    """
    return PointInput(vec=params.v_d)

#### Vector field controller #####
class VectorParams:
    """Parameters required for defining a constant velocity

    Attributes:
        v_max(float): Maximum velocity
    """
    def __init__(self, v_max: float) -> None:
        self.v_max = v_max

def vector_control(time: float, # pylint: disable=unused-argument
                   state: TwoDArrayType, # pylint: disable=unused-argument
                   vec: TwoDimArray,
                   params: VectorParams) -> PointInput: # pylint: disable=unused-argument
    """Implements a constant control for the single integrator dynamics

    Args:
        time: clock time (not used)
        state: vehicle state (not used)
        vec: The vector to be followed
        params: defines the maximum allowable velocity
    """
    # Throttle the vector to have the maximum velocity respected
    vec_cmd = vec.state
    mag = np.linalg.norm(vec_cmd)
    if mag > params.v_max:
        vec_cmd = vec_cmd * (params.v_max/mag)

    # Command the resulting velocity
    return PointInput(vec=TwoDimArray(vec=vec_cmd))
