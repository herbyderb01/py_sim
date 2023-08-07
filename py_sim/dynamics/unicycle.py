"""unicycle.py: Defines types, functions, and classes for the dynamics of a unicycle robot
"""

from typing import Protocol

import numpy as np
from py_sim.tools.sim_types import (
    ArcParams,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
    UnicycleStateProtocol,
    UnicyleControlProtocol,
)


class UnicycleParams:
    """The UnicycleParams class defines saturates for the inputs (i.e., max velocities)

    Attributes:
        v_max(float): The maximum allowed translational velocity
        w_max(float): The maximum allowed rotational velocity
    """
    def __init__(self, v_max: float = np.inf, w_max: float = np.inf) -> None:
        self.v_max = v_max
        self.w_max = w_max

def dynamics(state: UnicycleStateProtocol,
             control: UnicyleControlProtocol,
             params: UnicycleParams) -> UnicycleState:
    """ Calculates the dynamics of the unicycle.

    Note that even though a "UnicycleState" is returned, that actually corresponds to
    the time derivative
            d/dt x = v cos(psi)
            d/dt y = v sin(spi)
            d/dt psi = w

    Args:
        state: The current state of the vehicle
        control: The current input of the vehicle

    Returns:
        UnicycleState: The resulting state time derivative
    """
    # Saturate the velocities
    v = np.min([control.v, params.v_max]) # Saturate translational velocity
    v = np.max([control.v, -params.v_max])
    w = np.min([control.w, params.w_max]) # Saturate rotational velocity
    w = np.max([control.w, -params.w_max])

    state_dot = UnicycleState() # Time derivative of the state
    state_dot.x = v * np.cos(state.psi)
    state_dot.y = v * np.sin(state.psi)
    state_dot.psi = w
    return state_dot

###########################  Basic unicycle controllers ##################################
def velocity_control(time: float, # pylint: disable=unused-argument
                     state: UnicycleStateProtocol, # pylint: disable=unused-argument
                     vd: float,
                     wd: float) -> UnicycleControl: # pylint: disable=unused-argument
    """Implements a velocity control for the unicycle, which is simply copying the desired inputs

    Args:
        time: clock time (not used)
        state: vehicle state (not used)
        vd: Desired translational velocity
        wd: Desired rotational velocity

    Returns:
        UnicycleControl: Commanded velocities
    """
    return UnicycleControl(v=vd, w=wd)

def arc_control(time: float, # pylint: disable=unused-argument
                state: UnicycleStateProtocol,
                dyn_params: UnicycleParams, # pylint: disable=unused-argument
                cont_params: ArcParams) -> UnicycleControl:
    """ Implements the control for a circular arc

    Args:
        time: clock time
        state: vehicle state
        dyn_params: The parameters for the dynamics
        cont_params: The paramters for the control law

    Returns:
        UnicycleControl: Commanded velocities
    """
    return velocity_control(time=time, state=state, vd=cont_params.v_d, wd=cont_params.w_d)


######################### Velocity Based Vector Controllers ################################
class UniVelVecParams():
    """Parameters used for following a vector field

    Attributes:
        vd_field_max(float): Maximum desired velocity
        k_wd(float): gain on the error in the desired rotation angle
    """
    def __init__(self,
                vd_field_max: float = 5.,
                k_wd: float = 2.
                ) -> None:
        """Create the parameters

        Args:
            vd_field_max: Maximum desired velocity
            k_wd: gain on the error in the desired rotation angle
        """
        self.vd_field_max = vd_field_max
        self.k_wd = k_wd

class UniVelVecParamsProto(Protocol):
    """Defines parameters needed for unicycle velocity vector calculation

    Attributes:
        vd_field_max(float): Maximum desired velocity
        k_wd(float): gain on the error in the desired rotation angle
    """
    vd_field_max: float # Maximum desired velocity
    k_wd: float         # Gain on the error in the desired rotation angle

def desired_vector_follow_velocities(state: UnicycleStateProtocol,
                                     vec: TwoDimArray,
                                     params: UniVelVecParamsProto) -> tuple[float, float]:
    """ Calculates desired translational and rotational velocities to follow a vector field

    Args:
        time: The time for which the control is being calculated
        state: The state of the unicycle
        vec: The vector to be followed
        params: Parameters defining the control

    Returns:
        tuple[float, float]: returns (vd, wd) where vd is desired translational velocity and
        wd is the desired rotational velocity
    """

    # Calculate the desired velocity
    vd = np.linalg.norm(vec.state)
    vd = np.min([vd, params.vd_field_max]) # Threshold the desired velocity

    # Calculate the desired orientation
    psi_d = np.arctan2(vec.y, vec.x)

    # Calculate the error in orientation
    psi_e = state.psi-psi_d
    psi_e = np.arctan2(np.sin(psi_e), np.cos(psi_e)) # Adjust to ensure between -pi and pi

    # Calculate the desired rotational velocity
    wd = -params.k_wd*psi_e

    return (vd, wd)

def velocity_vector_field_control(time: float,  # pylint: disable=unused-argument
                               state: UnicycleStateProtocol,
                               vec: TwoDimArray,
                               dyn_params: UnicycleParams, # pylint: disable=unused-argument
                               cont_params: UniVelVecParamsProto) -> UnicycleControl:
    """velocity_vector_field_control will calculate the desired control to follow a vector field with the following inputs:

    Args:
        time: The time for which the control is being calculated
        state: The state of the unicycle
        vec: The vector to be followed
        dyn_params: The parameters for the dynamics
        cont_params: The paramters for the control law

    Returns:
        UnicycleControl: The commanded translational and rotational velocity
    """

    # Calculate the desired velocities
    vd, wd = desired_vector_follow_velocities(state=state, vec=vec, params=cont_params)

    # Use velocity control to follow the vector field
    return velocity_control(time=time, state=state, vd=vd, wd=wd)
