"""grid_visualization.py: Provides a framework for visualization the occupancy grid
"""

import copy
from typing import Generic

import numpy as np
import py_sim.sensors.occupancy_grid as og
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.plotting.plotting import PlotManifest
from py_sim.sim.generic_sim import SimParameters, start_sim
from py_sim.sim.sim_modes import SingleAgentSim
from py_sim.tools.sim_types import UnicycleControl, UnicycleState, UnicycleStateType
from py_sim.worlds.polygon_world import PolygonWorld, generate_world_obstacles


class GridVisualization(Generic[UnicycleStateType], SingleAgentSim[UnicycleStateType]):
    """Framework for implementing a simulator that just tests out a feedback controller"""
    def __init__(self,  # pylint: disable=too-many-arguments
                n_inputs: int,
                plots: PlotManifest[UnicycleStateType],
                world: PolygonWorld,
                params: SimParameters[UnicycleStateType]
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

        Args:
            initial_state: The starting state of the vehicle
            dynamics: The dynamics function to be used for simulation
            controller: The control law to be used during simulation
            control_params: The parameters of the control law to be used in simulation
            n_input: The number of inputs for the dynamics function
            params: The simulation parameters
        """

        # Initialize the parent SingleAgentSim class
        super().__init__(n_inputs=n_inputs, plots=plots, params=params)

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
    grid = og.generate_occupancy_from_polygon_world(world=obstacle_world,
                                                    res=0.25,
                                                    x_lim=(-5,25),
                                                    y_lim=(-5, 10))
    grid_inf = og.inflate_obstacles(grid=grid, inflation=0.5)

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 world=obstacle_world,
                                 #grid=grid,
                                 grid=grid_inf,
                                 plot_occupancy_grid=True,
                                 plot_occupancy_cells=False,
                                 plot_occupancy_circles=False)

    # Create the simulation
    params = SimParameters(initial_state=state_initial)
    params.sim_plot_period = 0.2
    params.sim_step = 0.1
    params.sim_update_period = 0.00001
    sim = GridVisualization(params=params,
                            n_inputs=UnicycleControl.n_inputs,
                            plots=plot_manifest,
                            world=obstacle_world)

    # Run the simulation
    start_sim(sim=sim)

if __name__ == "__main__":
    test_occupancy_grid()
