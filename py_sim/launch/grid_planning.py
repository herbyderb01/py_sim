"""grid_planning.py: Provides a framework for visualization of planning using an occupancy grid
"""

import time
from typing import Generic

import matplotlib.pyplot as plt
import py_sim.path_planning.forward_grid_search as search
from py_sim.sensors.occupancy_grid import generate_occupancy_from_polygon_world
from py_sim.sim.generic_sim import SingleAgentSim
from py_sim.tools.plot_constructor import create_plot_manifest
from py_sim.tools.plotting import PlotManifest
from py_sim.tools.sim_types import (
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
    UnicycleStateType,
)
from py_sim.worlds.polygon_world import PolygonWorld, generate_world_obstacles


class GridPlanning(Generic[UnicycleStateType], SingleAgentSim[UnicycleStateType]):
    """Framework for implementing a simulator that just tests out a feedback controller"""
    def __init__(self,  # pylint: disable=too-many-arguments
                initial_state: UnicycleStateType,
                n_inputs: int,
                plots: PlotManifest[UnicycleStateType],
                world: PolygonWorld,
                planner: search.ForwardGridSearch
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

            Inputs:
                initial_state: The starting state of the vehicle
                n_input: The number of inputs for the dynamics function
                plots: The plot manifest for creating the sim plots
                world: The polygon world used for planning
                planner: The path planner
        """

        super().__init__(initial_state=initial_state, n_inputs=n_inputs, plots=plots)

        # Initialize sim-specific parameters
        self.world: PolygonWorld = world
        self.planner: search.ForwardGridSearch = planner

    def update(self) -> None:
        """Does nothing - no movement of vehicle
        """

    def plan_update(self, plan_steps: int = -1) -> bool:
        """Makes a single update to the plan. Returns true if the planner
           has popped the goal off of the planning queue

            Inputs:
                plan_steps: The number of planning steps to take
        """
        # Determine the planning step
        steps = plan_steps
        if steps < 1:
            steps = self.planner.queue.count()
        for _ in range(steps):
            ind, succ = self.planner.step()
            if succ:
                break

        # Output results
        if ind < 0:
            print("No more possibilities for the planner")
            self.stop.set()
        if succ:
            print("Plan to goal found")
            self.stop.set()
        return succ

    def post_process(self) -> None:
        print("Finished planner")
        self.update_plot() # Plot the latest data

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

    # Create a planner
    ind_start, _ = grid.position_to_index(q=TwoDimArray(x = -2., y=-3.))
    ind_end, _ = grid.position_to_index(q=TwoDimArray(x = 14., y=7.))
    #planner = search.BreadFirstGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
    #planner = search.DepthFirstGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
    #planner = search.DijkstraGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
    planner = search.AstarGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)


    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 world=obstacle_world,
                                 plot_occupancy_grid=True,
                                 planner=planner
                                 )

    # Create the simulation
    sim = GridPlanning( initial_state=state_initial,
                        n_inputs=UnicycleControl.n_inputs,
                        plots=plot_manifest,
                        world=obstacle_world,
                        planner=planner)

    # Create a plan
    # if planner.search():
    #     ind_plan = planner.get_plan()
    #     print("Got plan: ", ind_plan)
    # else:
    #     print("Planning not successful")

    # Run the planning incrementally
    plt.show(block=False)
    finished = False # Flag indicating whether or not the planner has finished
    goal_found_advertised = False # Once the goal has been found, an message will be sent to terminal
    iteration = 0 # Keeps track of the number of planning iterations performed
    while not finished:
        # Display the iteration
        print("Plan iteration: ", iteration)
        iteration += 1

        # Update the plan and the resulting plot
        # (note that plan_steps=-1 will do a wave (as many steps as are in the queue) )
        finished = sim.plan_update(plan_steps=100)
        sim.update_plot()

        # Display if the goal has been found (note that planning will not stop until it is popped off the queue)
        if not goal_found_advertised:
            if ind_end in planner.parent_mapping:
                print("The goal has been found - continuing until popped")
                goal_found_advertised = True
        time.sleep(0.001)

    print('Planning finished, close figure')
    plt.show(block=True)

if __name__ == "__main__":
    test_occupancy_grid()
