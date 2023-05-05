"""simple_sim.py performs a test with a single vehicle"""

import asyncio
import copy
from typing import Any
import matplotlib.figure as mpl_fig
import matplotlib.axes._axes as mpl_ax
import matplotlib.animation as anim
import matplotlib.pyplot as plt
from threading import Lock, Thread
from py_sim.dynamics.unicycle import arc_control, dynamics
from py_sim.sim.generic_sim import (
    Data,
    SimParameters,
    Slice,
    euler_update,
    run_sim_simple,
)
from py_sim.tools.sim_types import ArcParams, UnicycleState
from py_sim.tools.plotting import initialize_position_plot, update_position_plot, OrientedPositionParams, init_oriented_position_plot, update_oriented_position_plot

StateType = UnicycleState

class SimpleSim():
    """Framework for implementing a simulator that just tests out a feedback controller"""
    def __init__(self) -> None:
        """Initialize the simulation
        """
        # Update the simulation parameters
        initial_state = StateType(x = 0., y= 0., psi= 0.)
        self.params = SimParameters[UnicycleState](initial_state=initial_state)
        self.control_params = ArcParams(v_d=1., w_d= 1.)

        # Create and store the data
        initial_slice: Slice[StateType] = Slice(state=self.params.initial_state, time=self.params.t0)
        self.data: Data[StateType] = Data(current=initial_slice)

        # Create a lock to store the data
        self.lock = Lock()

        # Create the figure and axis for plotting
        self.fig: mpl_fig.Figure
        self.ax: mpl_ax.Axes
        self.fig, self.ax = plt.subplots()
        self.position_plot = initialize_position_plot(ax=self.ax, label="Vehicle", color=(0.2, 0.36, 0.78, 1.0) )
        self.pose_params = OrientedPositionParams(rad=0.2)
        self.pose_plot = init_oriented_position_plot(ax=self.ax, params=self.pose_params)

    def setup(self) -> None:
        """Setup all of the storage and plotting"""
        # Initialize the plotting
        self.ax.set_title("Vehicle plot")
        self.ax.set_ylim(ymin=-2., ymax=2.)
        self.ax.set_xlim(xmin=-2., xmax=2.)

    async def update(self) -> None:
        """Calls all of the update functions
            * Gets the latest vector to be followed
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

        # Calculate the control to follow a circle
        control = arc_control(  time=self.data.current.time,
                                state=self.data.current.state,
                                params=self.control_params)

        # Update the state using the latest control
        self.data.next.state.state = euler_update(  dynamics=dynamics,
                                                    control=control,
                                                    initial=self.data.current.state,
                                                    dt=self.params.sim_step)


    def update_plot(self, _: Any) -> mpl_ax.Axes:
        """Plot the current values and state. Should be done with the lock on to avoid
           updating current while plotting the data
        """
        # Copy the state to avoid any conflicts
        with self.lock:
            plot_state = copy.deepcopy(self.data.current)

        # Update all of the plotting elements
        print("x = ", self.data.current.state.x, "y = ", self.data.current.state.y)
        update_position_plot(line=self.position_plot, location=plot_state.state)
        update_oriented_position_plot(plot=self.pose_plot, params=self.pose_params, pose=plot_state.state)
        return self.ax

    def continuous_plotting(self) -> None:
        """Plot the data at a certain rate"""
        print("Starting plot")
        self.ani = anim.FuncAnimation(self.fig, self.update_plot, interval=100)
        print("Showing")
        plt.show()

    async def post_process(self) -> None:
        """Process the results"""
        print("Final state: ", self.data.current.state.state)

def run_sim(sim: SimpleSim) -> None:
    """Runs the actual simulation of the data"""
    asyncio.run(run_sim_simple(sim=sim) )

if __name__ == "__main__":
    """Runs the simulation and the plotting"""
    # Run the simple simulation
    sim = SimpleSim()
    thread = Thread(target=run_sim, args=(sim,))
    thread.start()

    # Run the plotting
    sim.continuous_plotting()
