"""generic_sim.py: Defines a standard sim
"""

import asyncio
import copy
from typing import Any, Protocol

import numpy.typing as npt
from py_sim.tools.sim_types import Dynamics, Input, State


class SimParameters():
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
    tf: float = 10. # Final time of the simulation

    def __init__(self, initial_state: State) -> None:
        # Initial state of the vehicle
        self.initial_state: State = initial_state

class Slice():
    """ Contains a "slice" of data - the data produced / needed
        at a single time
    """
    def __init__(self, state: State, time: float = 0.) -> None:
        self.time = time # Current simulation time
        self.state: State = state # Current state

class Data():
    """Stores the changing simulation information"""
    def __init__(self, current: Slice) -> None:
        self.current = current # Stores the current slice of data to be read
        self.next = copy.deepcopy(current) # Stores the next data to be created

def euler_update(dynamics: Dynamics, initial: State, control: Input, dt: float) -> npt.NDArray[Any]:
    """Performs an eulers update to simulate forward one step on the dynamics

        Inputs:
            dynamics: a function handle for calculating the time derivative of the state
            initial: the starting state of the vehicle
            control: the control input to be applied
            dt: the time step of the update

        Outputs:
            The resulting time derivative of the state
    """
    result = initial.state + dt*( dynamics(initial, control).state )
    return result

class Sim(Protocol):
    """Basic class formulation for simulating"""
    data: Data # The simulation data
    params: SimParameters # Simulation parameters

    async def setup(self) -> None:
        """Setup all of the storage and plotting"""

    async def update(self) -> None:
        """Calls all of the update functions"""

    async def plot(self) -> None:
        """Plot the current values and state"""

    async def post_process(self) -> None:
        """Process the results"""




async def run_sim_simple(sim: Sim) -> None:
    """Run the simulation """

    # Setup the simulation
    await asyncio.gather(sim.setup())

    # Loop through the sim
    while sim.data.current.time <= sim.params.tf:
        await asyncio.gather(sim.update(), sim.plot())
        await asyncio.sleep(sim.params.sim_update_period)

    # Post process
    await asyncio.gather(sim.post_process())
