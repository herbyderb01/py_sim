"""grid_planning.py: Provides a framework for visualization of planning using an occupancy grid
"""

import copy
from typing import Generic

import numpy as np
from py_sim.sensors.occupancy_grid import generate_occupancy_from_polygon_world
from py_sim.sim.generic_sim import SingleAgentSim, start_simple_sim
from py_sim.tools.plot_constructor import create_plot_manifest
from py_sim.tools.plotting import PlotManifest
from py_sim.tools.sim_types import UnicycleControl, UnicycleState, UnicycleStateType
from py_sim.worlds.polygon_world import PolygonWorld, generate_world_obstacles

class GridPlanning(Generic[UnicycleStateType], SingleAgentSim[UnicycleStateType]):
    """Framework for implementing a simulator that just tests out a feedback controller"""
    def __init__(self,  # pylint: disable=too-many-arguments
                initial_state: UnicycleStateType,
                n_inputs: int,
                plots: PlotManifest[UnicycleStateType],
                world: PolygonWorld,
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

            Inputs:
                initial_state: The starting state of the vehicle
                dynamics: The dynamics function to be used for simulation
                controller: The control law to be used during simulation
                control_params: The parameters of the control law to be used in simulation
                n_input: The number of inputs for the dynamics function
        """

        super().__init__(initial_state=initial_state, n_inputs=n_inputs, plots=plots)

        # Initialize sim-specific parameters
        self.world: PolygonWorld = world

    def update(self) -> None:
        """Calls all of the update functions
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """
        # Calculate the control to follow the vector
        self.data.current.input_vec = np.zeros((2,1))

        # Update the state using the latest control
        self.data.next.state.state = copy.deepcopy(self.data.current.state.state)

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

def test_occupancy_grid() -> None:
    """Plots the occupancy grid on top of the world"""
    # Initialize the state and control
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)

    # Create the obstacle world and occupancy grid
    obstacle_world = generate_world_obstacles()
    grid = generate_occupancy_from_polygon_world(world=obstacle_world,
                                                 res=0.25,
                                                 x_lim=(-5,25),
                                                 y_lim=(-5, 10))

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 world=obstacle_world,
                                 grid=grid,
                                 plot_occupancy_grid=True)

    # Create the simulation
    sim = GridPlanning( initial_state=state_initial,
                        n_inputs=UnicycleControl.n_inputs,
                        plots=plot_manifest,
                        world=obstacle_world)

    # Update the simulation step variables
    sim.params.sim_plot_period = 0.1
    sim.params.sim_step = 0.1
    sim.params.sim_update_period = 0.00001
    start_simple_sim(sim=sim)

if __name__ == "__main__":
    test_occupancy_grid()
