"""Creates a simple test / visualization for trajectory tracking"""

from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import py_sim.dynamics.single_integrator as si
import py_sim.dynamics.unicycle as uni
from matplotlib.axes._axes import Axes
from py_sim.sim.integration import euler_update
from py_sim.tools.projections import calculate_line_distances
from py_sim.tools.sim_types import TwoDArrayType, TwoDimArray
from scipy.interpolate import interp1d, splev, splint, splprep


#############  Plot Assist Functions         ##########################
class PlotElement:
    """Stores the information of a line to plot"""
    def __init__(self,
                 label: str,
                 time: npt.NDArray[Any],
                 y: npt.NDArray[Any],
                 ax: Axes,
                 style: str,
                 x_label: str,
                 line_width: int
                 ) -> None:
        self.label = label
        self.time = time
        self.y = y
        self.ax = ax
        self.style = style
        self.x_label = x_label
        self.line_width = line_width

#############  General Trajectory Functions  ##########################
def interpolate_line(line: npt.NDArray[Any], s_vals: list[float], s_des: float) -> tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]:
    """Interpolates a line to find the position along the line of the desired spatial index

    Args:
        line: 2xn matrix where each column is a point in the line
        s_vals: n element list defining the spatial distance to each point along the line
        s_des: The desired spatial element to return

    Returns:
        tuple[npt.NDArray[Any], npt.NDArray[Any], npt.NDArray[Any]]: q, a, b
            q: The point along the line for s_des
            a: The line segment point before q
            b: The line segment point after q
    """
    # Check inputs
    if line.shape[0] != 2 or line.shape[1] < 2 or line.shape[1] != len(s_vals):
        raise ValueError("Invalid inputs")

    ### Special cases: s_des before or after the line
    # Case 1: s_des comes before any point on the line - move along first line segment
    if s_des <= s_vals[0]:
        q = line[:,[0]]
        a = line[:,[0]]
        b = line[:,[1]]
        return (q, a, b)

    # Case 2: s_des comes after any point on the line - move along last line segment
    if s_des >= s_vals[-1]:
        q = line[:, [-1]]
        a = line[:,[-2]]
        b = line[:,[-1]]
        return (q, a, b)

    ### Perform a linear search to find the indices of s_val that surround s_des
    ind_next = 0
    for (ind, s) in enumerate(s_vals):
        if s > s_des:
            ind_next = ind
            break

    # Calcuate the scaling of the point of interest
    s_prev = s_vals[ind_next - 1]
    s_next = s_vals[ind_next]
    scale = (s_des-s_prev) / (s_next - s_prev)

    ### Calculate the point of interest
    a = line[:, [ind_next-1]]
    b = line[:, [ind_next]]
    q = (1-scale)*a + scale*b
    return (q, a, b)

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
                 vel_des: float,
                 ds = 0.1):

        # Create a finer spline path
        s_vals = calculate_line_distances(line=spline_path)
        x_points: list[float] = []
        y_points: list[float] = []
        s = 0
        while s <= s_vals[-1]:
            q, _, _ = interpolate_line(line=spline_path, s_vals=s_vals, s_des=s)
            x_points.append(q.item(0))
            y_points.append(q.item(1))
            s += ds

        # #transfer x and y points of spline path to generate a splines path
        # x_points = spline_path[0,:].tolist()
        # y_points = spline_path[1,:].tolist()

        # Create the parametric spline
        #self.tck, _ = splprep([x_points, y_points], s=0.1, k=3, nest=30)
        self.tck, _ = splprep([x_points, y_points], s=1., nest=-1)

        # Integrate the distance
        dist = 0.
        x1, y1 = splev(0, self.tck, der=0)
        s = 0.
        while s < 1:
            # Calculate new point
            s += 0.01
            x2, y2 = splev(s, self.tck, der=0)

            # Calcualte the distance
            q1 = np.array([[x1], [y1]])
            q2 = np.array([[x2], [y2]])
            dist += float(np.linalg.norm(q1-q2))

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

class SegmentTraj:
    """Creates a trajectory from a piecewise continuous line

    Attributes:
        line(NDArray): 2xn, n > 1, matrix where each column is a point along the line
        s_vals(list[float]): list of n elements where each element is the distance along
                             the line to the point in self.line
        vel_des(float): The velocity (m/s) for the vehicle to travel
    """

    def __init__(self, line: npt.NDArray[Any], vel_des: float) -> None:
        if line.shape[0] != 2 or line.shape[1] < 2:
            raise ValueError("Line must be 2xn with n>1")
        self.line = line
        self.s_vals = calculate_line_distances(line=line)
        self.vel_des = vel_des


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
        # Get the desired spatial index and location
        s_des = self.vel_des * time
        (q, a, b) = interpolate_line(line=self.line, s_vals=self.s_vals, s_des=s_des)

        # Calculate the desired velocity
        u = (b-a) / np.linalg.norm(b-a)
        v = self.vel_des * u

        # Calculate the desired trajectory point
        q_d = np.concatenate((q, v), axis=1)
        return TrajPoint(q_d=q_d)

#############  Single Integrator Trajectory Functions #################
class TrajFollowParamsSI:
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

def traj_track_control_si(time: float, #pylint: disable=unused-argument
                          state: TwoDArrayType,
                          traj_point: TrajPoint,
                          cont_params: TrajFollowParamsSI) -> si.PointInput:
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

def test_trajectory_tracking_si() -> None:
    """Tests the trajectory tracking for a single integrator system
    """

    # Initialize the storage for the time, state, and inputs
    dt = 0.01
    t_vec = np.arange(start=0., stop=10., step=dt)
    len_t: int = t_vec.shape[0]
    x_mat = np.zeros(shape=(TwoDimArray.n_states, len_t))
    u_mat = np.zeros(shape=(si.PointInput.n_inputs, len_t-1))

    # Create the initial state
    x = TwoDimArray(x=-1., y=-1.)
    x_mat[:,[0]] = x.state
    si_params = si.SingleIntegratorParams()

    # Create the line to follow
    line = np.array([
        [1., 7., 13., 17.],
        [1., 3., 1.,  4.]
    ])
    vel_des = 1.5 # Desired velocity

    # Create the trajectory to follow
    # traj = SinusoidalTraj()
    # traj = SplineTraj(spline_path=line,
    #                   vel_des=vel_des)
    traj = SegmentTraj(line=line,
                       vel_des=vel_des)


    # Initialize trajectory values
    cont_params = TrajFollowParamsSI(K=1*np.identity(n=2))
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
        u_k = traj_track_control_si(time=t_k,
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
    _, ax = plt.subplots()

    # Plot the desired line to be followed
    if not isinstance(traj, SinusoidalTraj):
        ax.plot(line[0,:], line[1,:], '-g', linewidth=4, label="Line")

    # Plot the trajectory
    ax.plot(x_des_mat[0,:], x_des_mat[1,:], "-r", label="Desired", linewidth=3)
    ax.plot(x_mat[0,:], x_mat[1,:], "-b", label="Actual")
    ax.legend()
    ax.set_title("Trajectory")
    ax.set_aspect('equal', 'box')

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

    plt.show(block=False)
    plt.show()


#############  Unicycle Trajectory Functions #################
class DfcUniControl:
    """Control for the DFC

    Args:
        a(float): acceleration input
        w(float): rotational velocity input
    """
    def __init__(self, a: float, w: float) -> None:
        self.a = a
        self.w = w


class TrajFollowParamsDFC:
    """Parameters required for defining a trajectory following controller using the Dynamic Feedback Control (DFC) approach

    Attributes
        v_int(float): The integral of the acceleration command
        k_d(float): The gain on velocity error
        k_p(float): The gain on position error
    """
    def __init__(self, k_d: float, k_p: float, v_0: float) -> None:
        """ Initializes the dynamic feedback control parameters

        Args:
            k_d: Gain on velocity error
            k_p: gain on position error
            v_0: The initial velocity
        """
        self.v_int = v_0
        self.k_d = k_d
        self.k_p = k_p

def traj_track_control_dfc(time: float, #pylint: disable=unused-argument
                           state: uni.UnicycleStateProtocol,
                           traj_point: TrajPoint,
                           cont_params: TrajFollowParamsDFC) -> DfcUniControl:
    """Control used for trajectory tracking using the Dynamic Feedback Control (DFC) approach

    The control approach is taken from
        "Feedback control of a nonholonomic car-like robot" Chapter 3.3 subsection "Full state linearization via dynamic feedback"
    It has been adapted for unicycle kinematics

    Args:
        time: Time value of interest
        state: State at time
        traj_point: The desired trajectory point and its derivatives at the given time
        cont_params: The control parameters

    Returns:
        The control to as seen by the extended dynamics
    """
    # Calculate desired acceleration vector
    z = np.array([[state.x], [state.y]])
    zdot = np.array([[cont_params.v_int*np.cos(state.psi)],
                     [cont_params.v_int*np.sin(state.psi)]])
    r = traj_point.q_d_ddot - cont_params.k_d*(zdot- traj_point.q_d_dot) - cont_params.k_p*(z-traj_point.q_d)
    #r = -cont_params.k_d*(zdot- traj_point.q_d_dot) - cont_params.k_p*(z-traj_point.q_d)

    # print("q_d = \n", traj_point.q_d)
    # print("q_d_dot = \n", traj_point.q_d_dot)
    # print("q_d_ddot = \n", traj_point.q_d_ddot)

    # print("z = \n", z)
    # print("z_dot = \n", zdot)

    # Calculate the mapping from desired acceleration to the dfc output
    s = np.sin(state.psi)
    c = np.cos(state.psi)

    #v_inv = 1./np.max([1.e-6, cont_params.v_int])

    if cont_params.v_int > 0 and cont_params.v_int < 0.1:
        v_inv = 10.
    elif cont_params.v_int < 0 and cont_params.v_int > -1.e-6:
        v_inv = -10.
    else:
        v_inv = 1./cont_params.v_int



    M_inv = np.array([[c,       s],
                      [-v_inv*s, v_inv*c]])
    # M = np.array([[c, -cont_params.v_int*s],
    #               [s, cont_params.v_int*c]])
    # eye = M_inv@M
    # print("M_inv@M = \n", eye)

    # Calculate the desired output
    u = M_inv @ r
    return DfcUniControl(a=u.item(0), w=u.item(1))
    # a = -1.*(cont_params.v_int - 3)
    # return DfcUniControl(a = a, w=0.1)

def test_trajectory_tracking_uni_dfc() -> None:
    """Tests the Dynamic Feedback Control (DFC) for trajectory tracking developed in
        "Feedback control of a nonholonomic car-like robot" Chapter 3.3 subsection "Full state linearization via dynamic feedback"

        The approach has been adapted from a smooth bicycle model to a simple unicycle model
    """

    # Initialize the storage for the time, state, and inputs
    dt = 0.01
    t_vec = np.arange(start=0., stop=10., step=dt)
    len_t: int = t_vec.shape[0]
    x_mat = np.zeros(shape=(uni.UnicycleState.n_states, len_t))
    u_mat = np.zeros(shape=(si.PointInput.n_inputs, len_t-1))

    # Create the initial state
    x = uni.UnicycleState(x=-1., y=-1., psi=0.)
    x_mat[:,[0]] = x.state
    uni_params = uni.UnicycleParams(w_max=20.)

    # Create the line to follow
    line = np.array([
        [1., 7., 13., 17.],
        [1., 3., 1.,  4.]
    ])
    # line = np.array([
    #     [1., 70., 130., 17000.],
    #     [1., 30., 100.,  4000.]
    # ])
    vel_des = 1.5 # Desired velocity

    # Create the trajectory to follow
    # traj = SinusoidalTraj()
    traj = SplineTraj(spline_path=line,
                      vel_des=vel_des)
    # traj = SegmentTraj(line=line,
    #                    vel_des=vel_des)


    # Initialize trajectory values
    cont_params = TrajFollowParamsDFC(k_d=10.0, k_p=10., v_0=vel_des)
    x_des_mat = np.zeros(shape=(2,len_t))

    # Loop through and simulate the vehicle
    for k in range(len_t-1):
        # Extract state
        x_k = uni.UnicycleState(vec=x_mat[:,[k]])
        t_k = t_vec.item(k)


        # Calculate the desired trajectory point
        traj_point = traj.get_traj_point(time=t_k, der=2)
        x_des_mat[:, [k]] = traj_point.q_d

        # Calculate the input
        u_dfc = traj_track_control_dfc(time=t_k,
                                       state=x_k,
                                       traj_point=traj_point,
                                       cont_params=cont_params)
        u_k = uni.UnicycleControl(v=cont_params.v_int, w=u_dfc.w)
        u_mat[u_k.IND_V,k] = u_k.v
        u_mat[u_k.IND_W,k] = u_k.w

        # Update the state
        cont_params.v_int += dt*u_dfc.a
        x_kp1 = euler_update(dynamics=uni.dynamics,
                             initial=x_k,
                             control=u_k,
                             params=uni_params,
                             dt=dt)
        x_mat[:,[k+1]] = x_kp1

    # Store the final desired trajectory point
    traj_point = traj.get_traj_point(time=t_vec[-1], der=1)
    x_des_mat[:, [-1]] = traj_point.q_d

    ### Plot the results
    _, ax = plt.subplots()

    # Plot the desired line to be followed
    if not isinstance(traj, SinusoidalTraj):
        ax.plot(line[0,:], line[1,:], '-g', linewidth=4, label="Line")

    # Plot the trajectory
    ax.plot(x_des_mat[0,:], x_des_mat[1,:], "-r", label="Desired", linewidth=3)
    ax.plot(x_mat[0,:], x_mat[1,:], "-b", label="Actual")
    ax.legend()
    ax.set_title("Trajectory")
    ax.set_aspect('equal', 'box')

    # Plot the state elements
    _, ax_state = plt.subplots(nrows=5, ncols=1)
    ax_state[0].set_title("State Plots")
    ax_state[-1].set_xlabel("Time (sec)")


    # Plot the error plots
    _, ax_err = plt.subplots(nrows=2, ncols=1)
    ax_err[0].set_title("Error Plots")
    ax_err[-1].set_xlabel("Time (sec)")

    # Create the state plots
    plots : list[PlotElement] = []
    plots.append(PlotElement(label="u_1", time=t_vec[0:-1], y=u_mat[0,:],ax=ax_state[2], style='-b', x_label='u_1',line_width=1))
    plots.append(PlotElement(label="u_2", time=t_vec[0:-1],y=u_mat[1,:],ax=ax_state[3],style='-b',x_label='u_2',line_width=1))
    plots.append(PlotElement(label="x_d1",time=t_vec,y=x_des_mat[0,:],ax=ax_state[0],style='-r',x_label='x_1',line_width=3))
    plots.append(PlotElement(label="x_d2",time=t_vec,y=x_des_mat[1,:],ax=ax_state[1],style='-r',x_label='x_2',line_width=3))
    plots.append(PlotElement(label="x_1",time=t_vec,y=x_mat[0,:],ax=ax_state[0],style='-b',x_label='x_1',line_width=1))
    plots.append(PlotElement(label="x_2",time=t_vec,y=x_mat[1,:],ax=ax_state[1],style='-b',x_label='x_2',line_width=1))
    plots.append(PlotElement(label="e_1",time=t_vec,y=x_mat[0,:]-x_des_mat[0,:],ax=ax_err[0],style='-r',x_label='e_1',line_width=1))
    plots.append(PlotElement(label="e_2",time=t_vec,y=x_mat[1,:]-x_des_mat[1,:],ax=ax_err[1],style='-r',x_label='e_2',line_width=1))

    for plot in plots:
        plot.ax.plot(plot.time, plot.y, plot.style, label=plot.label, linewidth=plot.line_width)
        plot.ax.set_ylabel(plot.x_label)

    plt.show(block=False)
    plt.show()

if __name__ == "__main__":
    #test_trajectory_tracking_si()
    test_trajectory_tracking_uni_dfc()
