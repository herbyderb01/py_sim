"""generic_sim.py: Provides general data structures for the simulation
"""

import asyncio
import copy, time
from typing import Any, Generic, Protocol, cast

import numpy.typing as npt
from threading import Lock
from py_sim.tools.sim_types import Dynamics, InputType, StateType
import matplotlib.pyplot as plt


class SimParameters(Generic[StateType]):
    """ Contains a fixed set of parameters that do not change throughout
        the simulation, but are needed for simulation execution
    """
    # Update parameters
    sim_update_period: float = 0.1 # Period for successive calls to updating the simulation variables
    sim_plot_period: float = 1. # Period for which the plotting should occur

    # Timing parameters
    sim_step: float = 0.1 # Simulation time step (seconds), i.e., each call to the update() function is
                          # spaced by sim_step seconds in the simulation
    t0: float = 0. # The initial time of the simulation
    tf: float = 2. # Final time of the simulation

    def __init__(self, initial_state: StateType) -> None:
        # Initial state of the vehicle
        self.initial_state: StateType = initial_state

class Slice(Generic[StateType]):
    """ Contains a "slice" of data - the data produced / needed
        at a single time
    """
    def __init__(self, state: StateType, time: float = 0.) -> None:
        self.time = time # Current simulation time
        self.state: StateType = state # Current state

class Data(Generic[StateType]):
    """Stores the changing simulation information"""
    def __init__(self, current: Slice[StateType]) -> None:
        self.current = current # Stores the current slice of data to be read
        self.next = copy.deepcopy(current) # Stores the next data to be created

def euler_update(dynamics: Dynamics[StateType, InputType], initial: StateType, control: InputType, dt: float) -> npt.NDArray[Any]:
    """Performs an eulers update to simulate forward one step on the dynamics

        Inputs:
            dynamics: a function handle for calculating the time derivative of the state
            initial: the starting state of the vehicle
            control: the control input to be applied
            dt: the time step of the update

        Outputs:
            The resulting time derivative of the state
    """
    result = cast(npt.NDArray[Any], initial.state + dt*( dynamics(initial, control).state ))
    return result

class Sim(Protocol[StateType]):
    """Basic class formulation for simulating"""
    data: Data[StateType] # The simulation data
    params: SimParameters[StateType] # Simulation parameters
    lock: Lock # Lock for thread safe plotting

    def setup(self) -> None:
        """Setup all of the storage and plotting"""

    async def update(self) -> None:
        """Calls all of the update functions"""

    async def post_process(self) -> None:
        """Process the results"""

    def continuous_plotting(self) -> None:
        """Create a plot callback function
        """

async def run_sim_simple(sim: Sim[StateType]) -> None:
    """Run the simulation """

    # Setup the simulation
    sim.setup()
    #sim.continuous_plotting()

    # Loop through the sim
    while sim.data.current.time <= sim.params.tf:
        # Update the current state to be the previous next state
        with sim.lock:
            sim.data.current = copy.deepcopy(sim.data.next)
        print("t = ", sim.data.current.time)

        # Run the updates to calculate the new next state and plots
        await asyncio.gather(sim.update())
        await asyncio.sleep(sim.params.sim_update_period)


    # Post process
    await asyncio.gather(sim.post_process())
