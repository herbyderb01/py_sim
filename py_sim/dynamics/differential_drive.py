"""differential_drive.py defines types, functions, and classes for the dynamics of a differential drive robot
"""
from typing import Any, Optional, cast

import numpy as np
import numpy.typing as npt
import py_sim.dynamics.unicycle as uni
from py_sim.tools.sim_types import ArcParams, TwoDimArray, UnicycleState

DiffDriveState = UnicycleState

class DiffDriveParams:
    """Provides the parameters for a differential drive model

    Attributes:
        L(float): The length between the wheels
        R(float): The radius of the wheels
    """
    def __init__(self, L: float, R: float) -> None:
        self.L = L
        self.R = R

class DiffDriveControl:
    """Stores the inputs required for the differential drive dynamics (wheel rotational velocities)

    Attributes:
        IND_R(int): The index of the rotational velocity of the right wheel
        IND_L(int): The index of the rotational velocity of the left wheel
        input(NDArray[Any]): The input vector
        n_inputs(int): The number of inputs (i.e., rows in input)
        mu_l(float): The value of the left wheel rotational velocity
        mu_r(float): The value of the right wheel rotational velocity
    """
    IND_R: int = 0# The index of the rotational velocity of the right wheel
    IND_L: int = 1# The index of the rotational velocity of the left wheel
    n_inputs: int = 2 # The number of inputs (i.e., rows in input)

    def __init__(self, mu_r: float = 0., mu_l: float = 0., vec: Optional[npt.NDArray[Any]] = None) -> None:
        """Initializes the input vector from the passed in values. If vec is provided then mu_r and mu_l are ignored"""
        # Get the default values from the vector
        if vec is not None:
            mu_r = vec.item(self.IND_R)
            mu_l = vec.item(self.IND_L)

        # Create the input vector. Note that this avoids any issues with vec being a different shape
        # or modified elsewhere.
        self.input = np.array([[mu_r],[mu_l]])


    @property
    def mu_r(self) -> float:
        """Return the translational velocity"""
        return cast(float, self.input.item(self.IND_R))

    @mu_r.setter
    def mu_r(self, val: float) -> None:
        """Store the translational velocity"""
        self.input[self.IND_R,0] = val

    @property
    def mu_l(self) -> float:
        """Return the rotational velocity value"""
        return cast(float, self.input.item(self.IND_L))

    @mu_l.setter
    def mu_l(self, val: float) -> None:
        """Store the rotational velocity value"""
        self.input[self.IND_L,0] = val

def dynamics(state: DiffDriveState,
             control: DiffDriveControl,
             params: DiffDriveParams) -> DiffDriveState:
    """ Calculates the dynamics of the unicycle.

    Note that even though a "DiffDriveState" is returned, that actually corresponds to
    the time derivative
            d/dt x = v cos(psi)
            d/dt y = v sin(spi)
            d/dt psi = w

            where v = (mu_r + mu_l)/2 and w = (R/L)(mu_r-mu_l)

    Args:
        state: The current state of the vehicle
        control: The current input of the vehicle
        params: The vehicle parameters

    Returns:
        DiffDriveState: The resulting state time derivative
    """
    # Calculate the dynamics
    state_dot = DiffDriveState() # Time derivative of the state
    state_dot.x = 0.
    state_dot.y = 0.
    state_dot.psi = 0.
    return state_dot

###########################  Basic unicycle controllers ##################################
def velocity_control(time: float, state: DiffDriveState, dyn_params: DiffDriveParams, vd: float, wd: float) -> DiffDriveControl: # pylint: disable=unused-argument
    """Implements a velocity control for the unicycle, which is simply copying the desired inputs

    Args:
        time: clock time (not used)
        state: vehicle state (not used)
        vd: Desired translational velocity
        wd: Desired rotational velocity

    Returns:
        DiffDriveControl: The commanded wheel rotational velocities
    """
    return DiffDriveControl()

def arc_control(time: float,
                state: DiffDriveState,
                dyn_params: DiffDriveParams,
                cont_params: ArcParams) -> DiffDriveControl:
    """ Implements the control for a circular arc

    Args:
        time: clock time
        state: vehicle state
        params: control parameters

    Returns:
        DiffDriveControl: The commanded wheel rotational velocities
    """
    return velocity_control(time=time, state=state, dyn_params=dyn_params, vd=cont_params.v_d, wd=cont_params.w_d)

############################ Velocity Based Vector Controllers ######################################
def velocity_vector_field_control(time: float,
                               state: DiffDriveState,
                               vec: TwoDimArray,
                               dyn_params: DiffDriveParams,
                               cont_params: uni.UniVelVecParamsProto) -> DiffDriveControl:
    """velocity_vector_field_control will calculate the desired control to follow a vector field with the following inputs:

    Args:
        time: The time for which the control is being calculated
        state: The state of the unicycle
        vec: The vector to be followed
        params: Parameters defining the control

    Returns:
        DiffDriveControl: The commanded wheel rotational velocities
    """

    # Calculate the desired velocities
    vd, wd = uni.desired_vector_follow_velocities(state=state, vec=vec, params=cont_params)

    # Use velocity control to follow the vector field
    return velocity_control(time=time, state=state, dyn_params=dyn_params, vd=vd, wd=wd)
