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


class SingleIntegratorParams:
    """ Defines the attributes for the single integrator dynamics

    Attributes:
        v_max(NDArray[Any]): Maximum velocity of any individual component
    """
    def __init__(self, v_max: float = np.inf) -> None:
        self.v_max: npt.NDArray[Any] = np.array([[v_max],[v_max]])


def dynamics(state: TwoDArrayType, control: PointInput, params: SingleIntegratorParams) -> TwoDimArray: # pylint: disable=unused-argument
    """ Calculates single integrator dynamics.

    The returned time derivative corresponds to
        d/dt x
        d/dt y

    Args:
        state: The current position of the vehicle
        control: The current input (velocities)
        params: The parameters for the dynamics

    Returns:
        TwoDimArray: The resulting state time derivative (actually the input)
    """
    # Saturate the input
    vec: npt.NDArray[Any] = np.maximum(params.v_max, -params.v_max)
    vec: npt.NDArray[Any] = np.minimum(vec, params.v_max)

    # Give the command
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

def const_control(time: float, # pylint: disable=unused-argument
                  state: TwoDArrayType, # pylint: disable=unused-argument
                  dyn_params: SingleIntegratorParams, # pylint: disable=unused-argument
                  cont_params: ConstantInputParams) -> PointInput:
    """Implements a constant control for the single integrator dynamics

    Args:
        time: clock time (not used)
        state: vehicle state (not used)
        dyn_params: The parameters for the dynamics
        cont_params: The paramters for the control law
    """
    return PointInput(vec=cont_params.v_d)

######################### Velocity Based Vector Controllers ################################
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
                   dyn_params: SingleIntegratorParams, # pylint: disable=unused-argument
                   cont_params: VectorParams) -> PointInput: # pylint: disable=unused-argument
    """Implements a constant control for the single integrator dynamics

    Args:
        time: clock time (not used)
        state: vehicle state (not used)
        vec: The vector to be followed
        dyn_params: The parameters for the dynamics
        cont_params: The paramters for the control law
    """
    # Throttle the vector to have the maximum velocity respected
    vec_cmd = vec.state
    mag = np.linalg.norm(vec_cmd)
    if mag > cont_params.v_max:
        vec_cmd = vec_cmd * (cont_params.v_max/mag)

    # Command the resulting velocity
    return PointInput(vec=TwoDimArray(vec=vec_cmd))
