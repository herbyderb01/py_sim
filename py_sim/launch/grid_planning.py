"""grid_planning.py: Provides a framework for visualization of planning using an occupancy grid
"""

from typing import Generic

from py_sim.path_planning.forward_grid_search import (
    BreadFirstGridSearch,
    ForwardGridSearch,
)
from py_sim.sensors.occupancy_grid import generate_occupancy_from_polygon_world
from py_sim.sim.generic_sim import SingleAgentSim, start_simple_sim
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
                planner: ForwardGridSearch,
                plan_steps: int = -1
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

            Inputs:
                initial_state: The starting state of the vehicle
                n_input: The number of inputs for the dynamics function
                plots: The plot manifest for creating the sim plots
                world: The polygon world used for planning
                planner: The path planner
                plan_steps: The number of planning steps to take at each iteration
        """

        super().__init__(initial_state=initial_state, n_inputs=n_inputs, plots=plots)

        # Initialize sim-specific parameters
        self.world: PolygonWorld = world
        self.planner: ForwardGridSearch = planner
        self.plan_steps = plan_steps

    def update(self) -> None:
        """Calls all of the update functions
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """
        # Dummy calculations for state
        # self.data.current.input_vec = np.zeros((2,1)) # Calculate the control to follow the vector
        # self.data.next.state.state = copy.deepcopy(self.data.current.state.state) # Update the state using the latest control
        #self.data.next.time = self.data.current.time + self.params.sim_step # Update the time by sim_step

        # Take a planning step
        steps = self.plan_steps
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
    planner = BreadFirstGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 world=obstacle_world,
                                 grid=grid,
                                 plot_occupancy_grid=True,
                                 planner=planner
                                 )

    # Create the simulation
    sim = GridPlanning( initial_state=state_initial,
                        n_inputs=UnicycleControl.n_inputs,
                        plots=plot_manifest,
                        world=obstacle_world,
                        planner=planner,
                        plan_steps=-1)

    # Create a plan
    # if planner.search():
    #     ind_plan = planner.get_plan()
    #     print("Got plan: ", ind_plan)
    # else:
    #     print("Planning not successful")

    # Update the simulation step variables
    sim.params.sim_plot_period = 0.001
    sim.params.sim_step = 0.001 # irrelevant
    sim.params.sim_update_period = 0
    start_simple_sim(sim=sim)

if __name__ == "__main__":
    test_occupancy_grid()
