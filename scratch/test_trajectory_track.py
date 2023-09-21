"""Creates a simple test / visualization for trajectory tracking"""

import numpy as np
import numpy.typing as npt
from typing import Any
import matplotlib.pyplot as plt
import py_sim.dynamics.single_integrator as si
from py_sim.tools.sim_types import TwoDArrayType, TwoDimArray
from py_sim.sim.integration import euler_update
from scipy.interpolate import splev, splprep, splint

class TrajFollowParams:
    """Parameters required for defining a trajectory following controller

    Attributes:
        _K(NDArray): 2x2 matrix of for the feedback gain
    """
    def __init__(self, K: npt.NDArray[Any]) -> None:
        """ Input K must be a 2x2 matrix

        Args:
            K: Feedback control matrix
        """
        if K.shape[0] != 2 or K.shape[1] != 2:
            raise ValueError("Feedback control matrix must be 2x2")
        self._K = K

    @property
    def K(self) -> npt.NDArray[Any]:
        """Returns the control matrix"""
        return self._K

class TrajPoint:
    """Stores the desired point along a trajectory. Assumes derivative is zero for
       derivatives not specified at initialization

    Attributes:
        _q_d(NDArray): 2xn matrix where each column corresponds to a derivative wrt time
          [zeroth derivative, first, second, ...]
    """
    def __init__(self, q_d: npt.NDArray[Any]) -> None:
        """Initializes the trajectory storage, checks for correct size

        Args:
            q_d: 2xn matrix - Defines the desired position and its derivatives
                [zeroth derivative, first, second, ..., n-1 derivative]
        """
        if q_d.shape[0] != 2 or q_d.shape[1] < 1:
            raise ValueError("Incorrect shape for desired point")
        self._q_d = q_d

    @property
    def q_d(self) -> npt.NDArray[Any]:
        """Return the desired position"""
        return self._q_d[:,[0]]

    @property
    def q_d_dot(self) -> npt.NDArray[Any]:
        """Returns the desired velocity vector"""
        if self._q_d.shape[1] < 2:
            return np.zeros(shape=(2,1))
        return self._q_d[:,[1]]

    @property
    def q_d_ddot(self) -> npt.NDArray[Any]:
        """Returns the desired acceleration vector"""
        if self._q_d.shape[1] < 3:
            return np.zeros(shape=(2,1))
        return self._q_d[:,[2]]

    @property
    def q_d_dddot(self) -> npt.NDArray[Any]:
        """Returns the desired acceleration vector"""
        if self._q_d.shape[1] < 4:
            return np.zeros(shape=(2,1))
        return self._q_d[:,[3]]

    def derivative(self, der: int) -> npt.NDArray[Any]:
        """Returns the derivative of the state

        Args:
            der: The order of the derivative (must be >= 0)

        Returns:
            NDArray: The corresponding derivative of the state
        """
        if self._q_d.shape[1] < der+1:
            return np.zeros(shape=(2,1))
        return self._q_d[:, [der]]

class SinusoidalTraj:
    """Defines a basic sinusoidal trajectory"""
    def get_traj_point(self,
                       time: float,
                       der: int #pylint: disable=unused-argument
                       ) -> TrajPoint:
        """Return a trajectory point

        Args:
            time: Time value for the trajectory
            der: Maximum number of derivatives needed

        Returns:
            TrajPoint: The desired position and its time derivatives
        """
        q_d = np.array([
            #   q,          \dot{q}        \ddot{q}
            [time,          1.,            0.],
            [np.sin(time), np.cos(time), -np.sin(time)]
        ])
        return TrajPoint(q_d=q_d)

class SplineTraj:
    """Creates a trajectory from the spline points"""
    def __init__(self,
                 spline_path: npt.NDArray[Any],
                 vel_des: float):
        #transfer x and y points of spline path to generate a splines path
        x_points = spline_path[0,:].tolist()
        y_points = spline_path[1,:].tolist()

        # Create the parametric spline
        #self.tck, _ = splprep([x_points, y_points], s=0.1, k=3, nest=30)
        self.tck, _ = splprep([x_points, y_points], s=0, nest=-1)

        # Integrate the distance
        dist = 0
        x1, y1 = splev(0, self.tck, der=0)
        s = 0
        while s < 1:
            # Calculate new point
            s += 0.01
            x2, y2 = splev(s, self.tck, der=0)

            # Calcualte the distance
            q1 = np.array([[x1], [y1]])
            q2 = np.array([[x2], [y2]])
            dist += np.linalg.norm(q1-q2)

            # Update for next iteration
            x1 = x2
            y1 = y2

        # Determine the time to get to the final point
        spl_len, _ = splint(a=0, b=1, tck=self.tck) # Length of the spline
        #t_end = spl_len/vel_des
        t_end = dist/vel_des
        self.scale = 1./t_end

        print("Dist calc = ", dist, ", int = ", spl_len)

    def get_traj_point(self,
                       time: float,
                       der: int #pylint: disable=unused-argument
                       ) -> TrajPoint:
        """Return a trajectory point

        Args:
            time: Time value for the trajectory
            der: Maximum number of derivatives needed

        Returns:
            TrajPoint: The desired position and its time derivatives
        """
        s_val = time*self.scale

        q_d = np.zeros(shape=(2,der+1))
        for k in range(der+1):
            x,y = splev(s_val, self.tck, der=k)
            q_d[0,k] = x*self.scale**k
            q_d[1,k] = y*self.scale**k

        return TrajPoint(q_d=q_d)


def traj_track_control(time: float, #pylint: disable=unused-argument
                       state: TwoDArrayType,
                       traj_point: TrajPoint,
                       cont_params: TrajFollowParams) -> si.PointInput:
    """Implements a trajectory tracking controller

    Args:
        time: Time value of interest
        state: State at time
        traj_point: The desired trajectory point and its derivatives at the given time
        cont_params: The control parameters

    Returns:
        The control to move along the trajectory point
    """
    u = traj_point.q_d_dot - cont_params.K@(state.state - traj_point.q_d)
    return si.PointInput(vec=TwoDimArray(vec=u))

def test_trajectory_tracking() -> None:
    """Loops through and tests the trajectory tracking
    """

    # Initialize the storage for the time, state, and inputs
    dt = 0.01
    t_vec = np.arange(start=0., stop=10., step=dt)
    len_t: int = t_vec.shape[0]
    x_mat = np.zeros(shape=(TwoDimArray.n_states, len_t))
    u_mat = np.zeros(shape=(si.PointInput.n_inputs, len_t-1))

    # Create the initial state
    x = TwoDimArray(x=-1., y=1.)
    x_mat[:,[0]] = x.state
    si_params = si.SingleIntegratorParams()

    # Create teh trajectory to follow
    #traj = SinusoidalTraj()
    spline_path = np.array([
        [1., 10., 10., 20.],
        [0., 0.,  10., 10.]
    ])
    traj = SplineTraj(spline_path=spline_path,
                      vel_des=3.)

    # Initialize trajectory values
    cont_params = TrajFollowParams(K=1.*np.identity(n=2))
    x_des_mat = np.zeros(shape=(2,len_t))

    # Loop through and simulate the vehicle
    for k in range(len_t-1):
        # Extract state
        x_k = TwoDimArray(vec=x_mat[:,[k]])
        t_k = t_vec.item(k)


        # Calculate the desired trajectory point
        traj_point = traj.get_traj_point(time=t_k, der=1)
        x_des_mat[:, [k]] = traj_point.q_d

        # Calculate the input
        # u_k = si.PointInput(vec=TwoDimArray(
        #     x=np.cos(t_k),
        #     y=np.sin(t_k)
        # ))
        u_k = traj_track_control(time=t_k,
                                 state=x_k,
                                 traj_point=traj_point,
                                 cont_params=cont_params)
        u_mat[:,[k]] = u_k.vec.state

        # Update the state
        x_kp1 = euler_update(dynamics=si.dynamics,
                             initial=x_k,
                             control=u_k,
                             params=si_params,
                             dt=dt)
        x_mat[:,[k+1]] = x_kp1

    # Store the final desired trajectory point
    traj_point = traj.get_traj_point(time=t_vec[-1], der=1)
    x_des_mat[:, [-1]] = traj_point.q_d

    ### Plot the results
    # Plot the trajectory
    fig, ax = plt.subplots()
    ax.plot(x_des_mat[0,:], x_des_mat[1,:], "-r", label="desired", linewidth=3)
    ax.plot(x_mat[0,:], x_mat[1,:], "-b", label="state")
    ax.legend()
    ax.set_title("Trajectory")

    # Plot the state elements
    _, ax_state = plt.subplots(nrows=4, ncols=1)
    ax_state[0].set_title("State Plots")
    ax_state[-1].set_xlabel("Time (sec)")

    # Plot the error plots
    _, ax_err = plt.subplots(nrows=2, ncols=1)
    ax_err[0].set_title("Error Plots")
    ax_err[-1].set_xlabel("Time (sec)")

    # Create the state plots
    labels = ["u_1", "u_2", "x_d1", "x_d2", "x_1", "x_2", "e_1", "e_2"]
    times =  [t_vec[0:-1], t_vec[0:-1], t_vec, t_vec, t_vec, t_vec, t_vec, t_vec]
    y_vals = [u_mat[0,:], u_mat[1,:], x_des_mat[0,:], x_des_mat[1,:], x_mat[0,:], x_mat[1,:], x_mat[0,:]-x_des_mat[0,:], x_mat[1,:]-x_des_mat[1,:]]
    axes =   [ax_state[2], ax_state[3], ax_state[0], ax_state[1], ax_state[0], ax_state[1], ax_err[0], ax_err[1]]
    styles = ['-b', '-b', '-r', '-r', '-b', '-b', '-r', '-r']
    x_labels = ['u_1', 'u_2', 'x_1', 'x_2', 'x_1', 'x_2', "e_1", "e_2"]
    line_widths = [1, 1, 3, 3, 1, 1, 1, 1]
    for (label, time, y_val, ax, style, x_label, lw) in zip(labels, times, y_vals, axes, styles, x_labels, line_widths):
        ax.plot(time, y_val, style, label=label, linewidth=lw)
        ax.set_ylabel(x_label)

    plt.show()

if __name__ == "__main__":
    test_trajectory_tracking()
