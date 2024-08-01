"""generic_sim.py: Provides general data structures for the simulation
"""

import asyncio
import copy
import sys
from signal import SIGINT, signal
from threading import Event, Lock
from typing import Generic, Protocol

import matplotlib.pyplot as plt
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
    lock: Lock # Lock for updating the current state
    stop: Event # The sim should stop when the event is true

    def update(self) -> None:
        """Calls all of the update functions"""

    def post_process(self) -> None:
        """Process the results"""

    def update_plot(self) -> None:
        """Plot the current values and state. Should be done with the lock on to avoid
           updating current while plotting the data
        """
    def store_data_slice(self, sim_slice: Slice[StateType]) -> None:
        """Stores data after update"""

async def run_simulation(sim: Sim[StateType]) -> None:
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

async def continuous_plotting(sim: Sim[StateType]) -> None:
    """Plot the data at a certain rate"""
    # Create the initial plot
    plt.show(block=False)

    # Continuously update the plots
    while not sim.stop.is_set():
        sim.update_plot()
        await asyncio.sleep(sim.params.sim_plot_period)

    # Stop the simulator
    sim.stop.set()
    await asyncio.sleep(1.) # Allows for post processing to be started prior to blocking the thread
    print("Waiting for all plots to be closed")
    plt.show()

def start_sim(sim: Sim[StateType]) -> None:
    """Starts the simulation and the plotting of a simple sequential simulator

    Args:
        sim: The simulation to be run
    """

    async def run_sim(sim: Sim[StateType]) -> None:
        """Begins the async thread for running the simple sim"""
        await asyncio.gather(run_simulation(sim=sim), continuous_plotting(sim=sim))

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
