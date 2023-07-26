"""grid_navigation_function.py defines a navigation function using grid search techniques
"""

import numpy as np
from py_sim.tools.sim_types import (
    TwoDArrayType,
    TwoDimArray,
    VectorField,
)

from py_sim.worlds.polygon_world import PolygonWorld
from py_sim.sensors.occupancy_grid import generate_occupancy_from_polygon_world
import py_sim.path_planning.forward_grid_search as search

class GridNavigationFunction:
    """Defines a navigation function based upon a grid search
    """
    def __init__(self,
                 end: TwoDimArray,
                 obstacle_world: PolygonWorld,
                 plan_type: str,
                 v_des: float,
                 sig: float,
                 x_lim: tuple[float, float] = (-5., 25.),
                 y_lim: tuple[float, float] = (-5., 10.)) -> None:
        """ Creates the navigation function

        Args:
            start: The starting point for the plan
            end: The ending point for the plan
            obstacle_world: The world in which the planning is performed
            plan_type: identifies the planner to be used
                Grid-based planners: breadth, depth, dijkstra, astar, greedy
                Graph-based planners (both use dijkstra planning): visibility, voronoi
            v_des: Desired velocity for travel
            sig: the convergence factor for approaching zero velocity at the goal location
            x_lim: The limits on planning for the x-axis
            y_lim: The limits on planning for the y-axis
        """
        # Store class properties
        self.v_des = v_des
        self.sig_sq = sig**2
        self.planner: search.ForwardGridSearch

        # Grid-based planning
        if plan_type in ["breadth", "depth", "dijkstra", "astar", "greedy"]:
            # Create the obstacle world and occupancy grid
            grid = generate_occupancy_from_polygon_world(world=obstacle_world,
                                                        res=0.1,
                                                        x_lim=x_lim,
                                                        y_lim=y_lim)

            # Create a planner (note that the start and end are reversed so that the
            # node parent is the desired direction of travel)
            ind_end = 0 # ending index is not actually important
            ind_start, _ = grid.position_to_index(q=end)

            if plan_type == "breadth":
                self.planner = search.BreadFirstGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
            elif plan_type == "depth":
                self.planner = search.DepthFirstGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
            elif plan_type == "dijkstra":
                self.planner = search.DijkstraGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
            elif plan_type == "astar":
                self.planner = search.AstarGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
            elif plan_type == "greedy":
                self.planner = search.GreedyGridSearch(grid=grid, ind_start=ind_start, ind_end=ind_end)
            else:
                raise ValueError("Invalid grid-based planning type passed in")

            # Create the mapping for the entire grid
            self.planner.complete_search()

        else:
            raise ValueError("Invalid plan type")

    def calculate_vector(self, state: TwoDArrayType, time: float = 0.) -> TwoDimArray: # pylint: disable=unused-argument
        """Calculates a vector based upon the navigation function produced from a grid search

        Args:
            state: State of the vehicle
            time: Time of the state

        Returns:
            TwoDimArray: Vector pointing towards the goal
        """
        # Get the indices of the current cell and the cell's parent
        q_curr = state.position
        ind_curr, _ = self.planner.grid.position_to_index(q=TwoDimArray(vec=q_curr))
        try:
            ind_parent = self.planner.parent_mapping[ind_curr]
        except KeyError:
            ind_parent = self.planner.ind_start

        # Get the vector pointing from the current state to its parent
        q_parent = self.planner.grid.index_to_position(ind=ind_parent).position
        g = q_parent - q_curr

        # Calcualte the desired velocity based on the goal location
        q_goal = self.planner.grid.index_to_position(ind=self.planner.ind_start).position
        dist_to_goal = np.linalg.norm(q_curr - q_goal)
        v_g = self.v_des * (1.-np.exp(-dist_to_goal**2/self.sig_sq))

        # Scale the magnitude of the resulting vector
        dist = np.linalg.norm(g)
        if dist > 0.:
            g = (v_g/dist)*g
            result = TwoDimArray(vec=g)
        else:
            result = TwoDimArray(x=0., y=0.)
        return result