"""simple_sim.py performs a test with a single vehicle"""

import asyncio
import time
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
        (self.position_plot,) = self.ax.plot([0.], [0.], 'o', label='Vehicle', color=(0.2, 0.36, 0.78, 1.0) )

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
        with self.lock:
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


    def update_plot(self, _) -> mpl_ax.Axes:
        """Plot the current values and state"""
        with self.lock:
            print("x = ", self.data.current.state.x, "y = ", self.data.current.state.y)
            self.position_plot.set_data([self.data.current.state.x], [self.data.current.state.y])
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
