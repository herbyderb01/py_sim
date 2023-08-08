"""generic_sim.py: Provides general data structures for the simulation
"""

import asyncio
import copy
import sys
from signal import SIGINT, signal
from threading import Event, Lock
from typing import Generic, Protocol

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes._axes import Axes
from matplotlib.figure import Figure
from py_sim.plotting.plotting import DataPlot, PlotManifest, StatePlot
from py_sim.tools.sim_types import Data, Slice, StateType


class SimParameters(Generic[StateType]):
    """ Contains a fixed set of parameters that do not change throughout
        the simulation, but are needed for simulation execution

    Attributes:
        initial_state(StateType): State at the beginning of the simulation
        sim_update_period(float): Period for successive calls to updating the simulation variables
        sim_plot_period(float): Period for which the plotting should occur
        sim_step(float): Simulation time step (seconds)
        t0(float): The initial time of the simulation
        tf(float): Final time of the simulation
    """
    def __init__(self, initial_state: StateType) -> None:
        # Initial state of the vehicle
        self.initial_state: StateType = initial_state

        # Update parameters
        self.sim_update_period: float = 0.1 # Period for successive calls to updating the simulation variables
        self.sim_plot_period: float = 1. # Period for which the plotting should occur

        # Timing parameters
        self.sim_step: float = 0.1 # Simulation time step (seconds), i.e., each call to the update() function is
                            # spaced by sim_step seconds in the simulation
        self.t0: float = 0. # The initial time of the simulation
        self.tf: float = 20. # Final time of the simulation

class Sim(Protocol[StateType]):
    """Basic class formulation for simulating

    Attributes:
        data(Data[StateType]): The simulation data
        params(SimParameters[StateType]): Simulation parameters
        lock(Lock): Lock for thread safe plotting
        stop(Event): The sim should stop when the event is true

    """
    data: Data[StateType] # The simulation data
    params: SimParameters[StateType] # Simulation parameters
    lock: Lock # Lock for thread safe plotting
    stop: Event # The sim should stop when the event is true

    def update(self) -> None:
        """Calls all of the update functions"""

    def post_process(self) -> None:
        """Process the results"""

    async def continuous_plotting(self) -> None:
        """Create a plot callback function
        """

    def store_data_slice(self, sim_slice: Slice[StateType]) -> None:
        """Stores data after update"""

class SingleAgentSim(Generic[StateType]):
    """Implements the main functions for a single agent simulation

    Attributes:
        params(SimParameters): parameters for running the simulation
        data(Data): Stores the current and next slice of information
        lock(Lock): Lock used for writing to the data
        stop(Event): Event used to indicate that the simulator should be stopped
        figs(list[Figure]): Stores the figures that are used for plotting
        axes(dict[Axes, Figure]): Stores the axes used for plotting and their corresponding figures
        state_plots(list[StatePlot[StateType]]): Plots depending solely on state
        data_plots(list[DataPlot[StateType]]): Plots that depend on the data
    """
    def __init__(self,
                n_inputs: int,
                plots: PlotManifest[StateType],
                params: SimParameters[StateType]
                ) -> None:
        """Initialize the simulation
        """
        # Update the simulation parameters
        self.params = params

        # Create and store the data
        initial_slice: Slice[StateType] = Slice(state=self.params.initial_state, time=self.params.t0)
        self.data: Data[StateType] = Data(current=initial_slice)

        # Create a lock to store the data
        self.lock = Lock()

        # Create an event to stop the simulator
        self.stop = Event()

        # Initialize data storage
        self.initialize_data_storage(n_inputs=n_inputs)

        # Create the figure and axis for plotting
        self.figs: list[Figure] = plots.figs
        self.axes: dict[Axes, Figure] = plots.axes
        self.state_plots: list[StatePlot[StateType]] = plots.state_plots
        self.data_plots: list[DataPlot[StateType]] = plots.data_plots

    def initialize_data_storage(self, n_inputs: int) -> None:
        """Initializes all of the storage

        Args:
            n_inputs: The number of inputs for the control trajectory
        """
        num_elements_traj: int = int( (self.params.tf - self.params.t0)/self.params.sim_step ) + 2
            # Number of elements in the trajectory + 2 for the start and end times
        self.data.state_traj = np.zeros((self.data.current.state.n_states, num_elements_traj))
        self.data.time_traj = np.zeros((num_elements_traj,))
        self.data.control_traj = np.zeros((n_inputs, num_elements_traj))
        self.data.traj_index_latest = -1 # -1 indicates that nothing has yet been saved

    def update(self) -> None:
        """Performs all the required updates"""
        raise NotImplementedError("Update function must be implemented")

    def update_plot(self) -> None:
        """Plot the current values and state. Should be done with the lock on to avoid
           updating current while plotting the data
        """
        # Copy the state to avoid any conflicts
        with self.lock:
            plot_state = copy.deepcopy(self.data.current)

        # Update all of the plotting elements
        for plotter in self.state_plots:
            plotter.plot(state=plot_state.state)

        # Update all of the data plotting elements
        for plotter in self.data_plots:
            plotter.plot(data=self.data)

        # Flush all of the figures
        for fig in self.figs:
            fig.canvas.draw()
            fig.canvas.flush_events()

    def store_data_slice(self, sim_slice: Slice[StateType]) -> None:
        """Stores the state trajectory data

        Args:
            sim_slice: The information to be stored
        """
        with self.lock:
            # Check size - double if insufficient
            if self.data.traj_index_latest+1 >= self.data.state_traj.shape[1]: # Larger than allocated
                self.data.state_traj = np.append(self.data.state_traj, \
                    np.zeros(self.data.state_traj.shape), axis=1 )
                self.data.time_traj = np.append(self.data.time_traj, np.zeros(self.data.time_traj.size))
                self.data.control_traj = np.append(self.data.control_traj,
                                                   np.zeros(self.data.control_traj.shape),
                                                   axis=1)

            # Store data
            self.data.traj_index_latest += 1
            self.data.state_traj[:,self.data.traj_index_latest:self.data.traj_index_latest+1] = \
                sim_slice.state.state
            self.data.time_traj[self.data.traj_index_latest] = sim_slice.time

            if sim_slice.input_vec is not None:
                self.data.control_traj[:,self.data.traj_index_latest:self.data.traj_index_latest+1] = \
                sim_slice.input_vec

    async def continuous_plotting(self) -> None:
        """Plot the data at a certain rate"""
        # Create the initial plot
        plt.show(block=False)

        # Continuously update the plots
        while not self.stop.is_set():
            self.update_plot()
            await asyncio.sleep(self.params.sim_plot_period)

        # Stop the simulator
        self.stop.set()
        await asyncio.sleep(1.) # Allows for post processing to be started prior to blocking the thread
        print("Waiting for all plots to be closed")
        plt.show()

    def post_process(self) -> None:
        """Process the results"""
        print("Final state: ", self.data.current.state.state)
        print("State trajectory: ", self.data.state_traj)
        # print("Time trajectory: ", self.data.time_traj[0:self.data.traj_index_latest+1])

async def run_sim_simple(sim: Sim[StateType]) -> None:
    """Run the simulation """
    # Loop through the sim
    while sim.data.current.time <= sim.params.tf and not sim.stop.is_set():
        # Update the current state to be the previous next state
        with sim.lock:
            sim.data.current = copy.deepcopy(sim.data.next)

        # Run the updates to calculate the new next state and plots
        sim.update()
        sim.store_data_slice(sim.data.current)

        await asyncio.sleep(sim.params.sim_update_period)
    sim.store_data_slice(sim.data.next) # Store the final data
    sim.stop.set()


    # Post process
    print("Post-processing")
    sim.post_process()

def start_simple_sim(sim: Sim[StateType]) -> None:
    """Starts the simulation and the plotting of a simple sequential simulator

    Args:
        sim: The simulation to be run
    """

    async def run_sim(sim: Sim[StateType]) -> None:
        """Begins the async thread for running the simple sim"""
        await asyncio.gather(run_sim_simple(sim=sim), sim.continuous_plotting())

    def handler(_, __): # type: ignore
        """Simple signal handler"""
        # Handle any cleanup here
        if sim.stop.is_set():
            print('Second stop signal detected, exiting')
            sys.exit(0)

        # Stop the sim
        print('Stopping sim - running post processing. Close figures or press ctrl-c again to exit')
        sim.stop.set()

    signal(SIGINT, handler=handler) # type: ignore
    asyncio.run(run_sim(sim=sim))
