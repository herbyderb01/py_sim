"""grid_planning.py: Provides a framework for visualization of planning using an occupancy grid

Basic graph search techniques are implemented, including:
    * Breadth-first
    * Depth-first
    * Dijkstra
    * A*
    * Greedy
"""

import time
from typing import Generic

import matplotlib.pyplot as plt
import py_sim.path_planning.forward_grid_search as search
import py_sim.worlds.polygon_world as poly_world
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.plotting.plotting import PlotManifest
from py_sim.sensors.occupancy_grid import generate_occupancy_from_polygon_world
from py_sim.sim.generic_sim import SimParameters
from py_sim.sim.sim_modes import SingleAgentSim
from py_sim.tools.sim_types import (
    Data,
    Slice,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
    UnicycleStateType,
)


class GridPlanning(Generic[UnicycleStateType], SingleAgentSim[UnicycleStateType]):
    """Framework for implementing a simulator that just tests out a feedback controller

    Attributes:
        world(PolygonWorld): World in which the planning occurs
        planner(ForwardGridSearch): The planner that is used for planning
    """
    def __init__(self,  # pylint: disable=too-many-arguments
                n_inputs: int,
                plots: PlotManifest[UnicycleStateType],
                world: poly_world.PolygonWorld,
                planner: search.ForwardGridSearch,
                params: SimParameters[UnicycleStateType]
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

        Args:
            n_input: The number of inputs for the dynamics function
            plots: The plot manifest for creating the sim plots
            world: The polygon world used for planning
            planner: The path planner
            params: The paramters used for simulation
        """

        # Create the data storage
        initial_slice: Slice[UnicycleStateType] = Slice(state=params.initial_state, time=params.t0)
        data: Data[UnicycleStateType] = Data(current=initial_slice)

        # Initialize the parent SingleAgentSim class
        super().__init__(n_inputs=n_inputs, plots=plots, params=params, data=data)

        # Initialize sim-specific parameters
        self.world: poly_world.PolygonWorld = world
        self.planner: search.ForwardGridSearch = planner

    def update(self) -> None:
        """Does nothing - no movement of vehicle
        """

    def plan_update(self, plan_steps: int = -1) -> bool:
        """Makes a single update to the plan. Returns true if the planner
           has popped the goal off of the planning queue

        Args:
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
        """Final update on the plot after the simulation has ended
        """
        print("Finished planner")
        self.update_plot() # Plot the latest data

def test_grid_planner() -> None:
    """ Plans a path using a grid-based approach

       Grid path planning done in the following steps:
        * Create a world
        * Create a planner
        * Create the plotting
        * Incrementally calculate the plan until plan calculated
        * Calculate the plan length
    """
    # Initialize the state and control
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)

    # Create the obstacle world
    obstacle_world = poly_world.generate_world_obstacles()
    # obstacle_world = poly_world.generate_non_convex_obstacles()

    # Create the occupancy grid from the world
    grid = generate_occupancy_from_polygon_world(world=obstacle_world,
                                                 res=0.25,
                                                 x_lim=(-5,25),
                                                 y_lim=(-5, 10))

    # Create the starting and stopping indices for the planning
    ind_start, _ = grid.position_to_index(q=TwoDimArray(x = -2., y=-3.))
    ind_end, _ = grid.position_to_index(q=TwoDimArray(x = 14., y=7.))

    # Create a planner
    #planner = search.BreadFirstGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
    #planner = search.DepthFirstGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
    planner = search.DijkstraGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
    #planner = search.AstarGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
    #planner = search.GreedyGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)


    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 world=obstacle_world,
                                 plot_occupancy_grid=True,
                                 planner=planner
                                 )

    # Create the simulation
    params = SimParameters(initial_state=state_initial)
    sim = GridPlanning( params=params,
                        n_inputs=UnicycleControl.n_inputs,
                        plots=plot_manifest,
                        world=obstacle_world,
                        planner=planner)

    ### Create a plan all at once (not incrementally) Uncomment this block to use and commend out the next block ###
    # if planner.search():
    #     ind_plan = planner.get_plan()
    #     print("Got plan: ", ind_plan)
    # else:
    #     print("Planning not successful")

    ### Create a plan incrementally (not all at once) - Useful for plotting - Comment out previous block to use ###
    # Create the plot and initialized iteration variables
    plt.show(block=False)
    finished = False # Flag indicating whether or not the planner has finished
    goal_found_advertised = False # Once the goal has been found, a message will be sent to terminal
    iteration = 0 # Keeps track of the number of planning iterations performed

    # Iteratively create a plan and visualize it
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

    ### Print out the plan length and close the figure ###
    # Calculate the plan length
    plan_length = planner.calculate_plan_length()
    print('Plan length = ', plan_length)

    print('Planning finished, close figure')
    plt.show(block=True)

if __name__ == "__main__":
    test_grid_planner()
