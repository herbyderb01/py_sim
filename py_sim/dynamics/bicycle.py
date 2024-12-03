"""bicycle.py defines types, functions, and classes for the dynamics of a bicycle robot
"""
from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt
import py_sim.dynamics.unicycle as uni
from py_sim.tools.sim_types import ArcParams, TwoDimArray, UnicycleState

BicycleState = UnicycleState

class BicycleParams:
    """Provides the parameters for a bicycle model

    Attributes:
        L(float): The length between the wheels
    """
    def __init__(self, L: float) -> None:
        self.L = L

class BicycleControl:
    """Stores the inputs required for the bicycle dynamics (translational velocity and turn angle)

    Attributes:
        IND_V(int): The index of the translational velocity
        IND_PHI(int): The index of the turn angle of the wheel
        input(NDArray[Any]): The input vector
        n_inputs(int): The number of inputs (i.e., rows in input)
        v(float): The value of the translational
        phi(float): The value of the turn angle
    """
    IND_V: int = 0# The index of the translational velocity
    IND_PHI: int = 1# The index of the turn angle of the wheel
    n_inputs: int = 2 # The number of inputs (i.e., rows in input)

    def __init__(self, v: float = 0., phi: float = 0., vec: Optional[npt.NDArray[Any]] = None) -> None:
        """Initializes the input vector from the passed in values. If vec is provided then v and phi are ignored"""
        # Get the default values from the vector
        if vec is not None:
            v = vec.item(self.IND_V)
            phi = vec.item(self.IND_PHI)

        # Create the input vector. Note that this avoids any issues with vec being a different shape
        # or modified elsewhere.
        self.input = np.array([[v],[phi]])


    @property
    def v(self) -> float:
        """Return the translational velocity"""
        return cast(float, self.input.item(self.IND_V))

    @v.setter
    def v(self, val: float) -> None:
        """Store the translational velocity"""
        self.input[self.IND_V,0] = val

    @property
    def phi(self) -> float:
        """Return the rotational velocity value"""
        return cast(float, self.input.item(self.IND_PHI))

    @phi.setter
    def phi(self, val: float) -> None:
        """Store the rotational velocity value"""
        self.input[self.IND_PHI,0] = val

def dynamics(state: BicycleState,
             control: BicycleControl,
             params: BicycleParams) -> BicycleState:
    """ Calculates the dynamics of the unicycle.

    Note that even though a "BicycleState" is returned, that actually corresponds to
    the time derivative
            d/dt x = v cos(psi)
            d/dt y = v sin(spi)
            d/dt psi = w

            where w = (v/L)(tan(phi))

    Args:
        state: The current state of the vehicle
        control: The current input of the vehicle
        params: The vehicle parameters

    Returns:
        BicycleState: The resulting state time derivative
    """
    v = control.v
    w = (v/params.L)*(np.tan(control.phi))

    state_d = BicycleState()
    state_d.x = v * np.cos(state.psi)
    state_d.y = v * np.sin(state.psi)
    state_d.psi = w
    return state_d

###########################  Basic unicycle controllers ##################################
def velocity_control(time: float, state: BicycleState, dyn_params: BicycleParams, vd: float, wd: float) -> BicycleControl: # pylint: disable=unused-argument
    """Implements a velocity control for the unicycle, which is simply copying the desired inputs

    Args:
        time: clock time (not used)
        state: vehicle state (not used)
        vd: Desired translational velocity
        wd: Desired rotational velocity

    Returns:
        BicycleControl: The commanded wheel rotational velocities
    """
    phi_d = np.arctan2(dyn_params.L*wd, vd)

    return BicycleControl(v=vd, phi=phi_d)

def arc_control(time: float,
                state: BicycleState,
                dyn_params: BicycleParams,
                cont_params: ArcParams) -> BicycleControl:
    """ Implements the control for a circular arc

    Args:
        time: clock time
        state: vehicle state
        params: control parameters

    Returns:
        BicycleControl: The commanded wheel rotational velocities
    """
    return velocity_control(time=time, state=state, dyn_params=dyn_params, vd=cont_params.v_d, wd=cont_params.w_d)

############################ Velocity Based Vector Controllers ######################################
def velocity_vector_field_control(time: float,
                               state: BicycleState,
                               vec: TwoDimArray,
                               dyn_params: BicycleParams,
                               cont_params: uni.UniVelVecParamsProto) -> BicycleControl:
    """velocity_vector_field_control will calculate the desired control to follow a vector field with the following inputs:

    Args:
        time: The time for which the control is being calculated
        state: The state of the unicycle
        vec: The vector to be followed
        params: Parameters defining the control

    Returns:
        BicycleControl: The commanded wheel rotational velocities
    """

    # Calculate the desired velocities
    vd, wd = uni.desired_vector_follow_velocities(state=state, vec=vec, params=cont_params)

    # Use velocity control to follow the vector field
    return velocity_control(time=time, state=state, dyn_params=dyn_params, vd=vd, wd=wd)
