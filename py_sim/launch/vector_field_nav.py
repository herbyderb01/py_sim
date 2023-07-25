"""vector_field_nav.py: Provides sample vector fields used for navigation
"""

from typing import Generic

from py_sim.dynamics import single_integrator
from py_sim.dynamics.unicycle import UniVelVecParams
from py_sim.dynamics.unicycle import dynamics as unicycle_dynamics
from py_sim.dynamics.unicycle import velocityVectorFieldControl
from py_sim.path_planning.path_generation import create_path
from py_sim.plotting.plot_constructor import create_plot_manifest
from py_sim.plotting.plotting import PlotManifest
from py_sim.sensors.range_bearing import RangeBearingSensor
from py_sim.sim.generic_sim import SingleAgentSim, start_simple_sim
from py_sim.sim.integration import euler_update
from py_sim.tools.sim_types import (
    ControlParamType,
    Dynamics,
    InputType,
    LocationStateType,
    TwoDimArray,
    UnicycleControl,
    UnicycleState,
    VectorControl,
)
from py_sim.vectorfield.vectorfields import G2GAvoid  # pylint: disable=unused-import
from py_sim.worlds.polygon_world import PolygonWorld, generate_world_obstacles


class NavVectorFollower(Generic[LocationStateType, InputType, ControlParamType], SingleAgentSim[LocationStateType]):
    """Framework for implementing a simulator that uses a vector field for feedback control through a polygon world with a distance measurement

    Attributes:
        dynamics(Dynamics[LocationStateType, InputType]): The dynamics function to be used for simulation
        controller(Control[LocationStateType, InputType, ControlParamType]): The control law to be used during simulation
        control_params(ControlParamType): The parameters of the control law to be used in simulation
        vector_field(G2GAvoid): Vector field that the vehicle will use to avoid obstacles while traversing to the goal
        world(PolygonWorld): World in which the vehicle is operating
        sensor(RangeBearingSensor): The sensor used for detecting obstacles
    """
    def __init__(self,  # pylint: disable=too-many-arguments
                initial_state: LocationStateType,
                dynamics: Dynamics[LocationStateType, InputType],
                controller: VectorControl[LocationStateType, InputType, ControlParamType],
                control_params: ControlParamType,
                n_inputs: int,
                plots: PlotManifest[LocationStateType],
                vector_field: G2GAvoid,
                world: PolygonWorld,
                sensor: RangeBearingSensor
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage

        Args:
            initial_state: The starting state of the vehicle
            dynamics: The dynamics function to be used for simulation
            controller: The control law to be used during simulation
            control_params: The parameters of the control law to be used in simulation
            n_input: The number of inputs for the dynamics function
        """

        super().__init__(initial_state=initial_state, n_inputs=n_inputs, plots=plots)

        # Initialize sim-specific parameters
        self.dynamics: Dynamics[LocationStateType, InputType] = dynamics
        self.controller: VectorControl[LocationStateType, InputType, ControlParamType] = controller
        self.control_params: ControlParamType = control_params
        self.vector_field: G2GAvoid = vector_field
        self.world: PolygonWorld = world
        self.sensor: RangeBearingSensor = sensor

    def update(self) -> None:
        """Calls all of the update functions.

        Updates performed:
            * Calculate the range and bearing measurements
            * Calculate the resulting vector to be followed
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """
        # Calculate the range bearing for the current position
        self.data.range_bearing_latest = self.sensor.calculate_range_bearing_measurement(\
            pose=self.data.current.state,
            world=self.world)

        # Calculate the desired vector
        self.vector_field.update_obstacles(locations=self.data.range_bearing_latest.location)
        vec: TwoDimArray = self.vector_field.calculate_vector(state=self.data.current.state, time=self.data.current.time)

        # Calculate the control to follow the vector
        control:InputType = self.controller(time=self.data.current.time,
                                state=self.data.current.state,
                                vec=vec,
                                params=self.control_params)
        self.data.current.input_vec = control.input

        # Update the state using the latest control
        self.data.next.state.state = euler_update(  dynamics=self.dynamics,
                                                    control=control,
                                                    initial=self.data.current.state,
                                                    dt=self.params.sim_step)

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

def run_unicycle_simple_vectorfield_example(follow_path: bool = False) -> None:
    """Runs an example of a go-to-goal vector field combined with obstacle avoidance to show off the sensor measurements being performed

    Args:
        follow_path: True => a path will be created and followed, False => the vector field will alone be used
                     for navigating to the goal
    """

    # Initialize the state and control
    vel_params = UniVelVecParams(vd_field_max=5., k_wd= 2.)
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)

    # Create the vector field
    n_lines = 10 # Number of sensor lines
    vector_field = G2GAvoid(x_g=TwoDimArray(x=10., y=5.),
                            n_obs=n_lines,
                            v_max=vel_params.vd_field_max,
                            S=1.5,
                            R=1.,
                            sig=1.)

    # Create the obstacle world
    obstacle_world = generate_world_obstacles()

    # Create the plan
    if follow_path:
        plan = create_path(start=TwoDimArray(x=state_initial.x, y=state_initial.y), end=vector_field.x_g, obstacle_world=obstacle_world, plan_type="voronoi")
    else:
        plan = None # No plan to follow

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 position_dot=False,
                                 position_triangle=True,
                                 state_trajectory=True,
                                 time_series=True,
                                 vectorfield=vector_field,
                                 vector_res=0.5,
                                 world=obstacle_world,
                                 range_bearing_locations=True,
                                 range_bearing_lines=True,
                                 plan=plan)

    # Create the simulation
    sim = NavVectorFollower(initial_state=state_initial,
                         dynamics=unicycle_dynamics,
                         controller=velocityVectorFieldControl,
                         control_params=vel_params,
                         n_inputs=UnicycleControl.n_inputs,
                         plots=plot_manifest,
                         vector_field=vector_field,
                         world=obstacle_world,
                         sensor=RangeBearingSensor(n_lines=n_lines, max_dist=4.)
                         )

    # Update the simulation step variables
    sim.params.sim_plot_period = 0.1
    sim.params.sim_step = 0.1
    sim.params.sim_update_period = 0.1
    start_simple_sim(sim=sim)

def run_single_simple_vectorfield_example(follow_path: bool = False) -> None:
    """Runs an example of a go-to-goal vector field combined with obstacle avoidance to show off the sensor measurements being performed using a single integrator

    Args:
        follow_path: True => a path will be created and followed, False => the vector field will alone be used
                     for navigating to the goal
    """
    # Initialize the state and control
    vel_params = single_integrator.VectorParams(v_max=5.)
    state_initial = TwoDimArray(x = 0., y= 0.)

    # Create the vector field
    n_lines = 10 # Number of sensor lines
    vector_field = G2GAvoid(x_g=TwoDimArray(x=10., y=5.),
                            n_obs=n_lines,
                            v_max=vel_params.v_max,
                            S=1.5,
                            R=1.,
                            sig=1.)

    # Create the obstacle world
    obstacle_world = generate_world_obstacles()

    # Create the plan
    if follow_path:
        plan = create_path(start=TwoDimArray(x=state_initial.x, y=state_initial.y), end=vector_field.x_g, obstacle_world=obstacle_world, plan_type="voronoi")
    else:
        plan = None # No plan to follow

    # Create the manifest for the plotting
    plot_manifest = create_plot_manifest(initial_state=state_initial,
                                 y_limits=(-5, 10),
                                 x_limits=(-5, 25),
                                 position_dot=True,
                                 state_trajectory=True,
                                 time_series=True,
                                 vectorfield=vector_field,
                                 vector_res=0.5,
                                 world=obstacle_world,
                                 range_bearing_locations=True,
                                 range_bearing_lines=True,
                                 plan=plan)

    # Create the simulation
    sim = NavVectorFollower(initial_state=state_initial,
                         dynamics=single_integrator.dynamics,
                         controller=single_integrator.vector_control,
                         control_params=vel_params,
                         n_inputs=single_integrator.PointInput.n_inputs,
                         plots=plot_manifest,
                         vector_field=vector_field,
                         world=obstacle_world,
                         sensor=RangeBearingSensor(n_lines=n_lines, max_dist=4.)
                         )

    # Update the simulation step variables
    sim.params.sim_plot_period = 0.1
    sim.params.sim_step = 0.1
    sim.params.sim_update_period = 0.1
    start_simple_sim(sim=sim)

if __name__ == "__main__":
    # Perform navigation without path planning (simple goal and avoid vector fields)
    #run_unicycle_simple_vectorfield_example(follow_path=False)
    #run_single_simple_vectorfield_example(follow_path=False)

    # Perform navigation with path planning using a carrot follower
    #run_unicycle_simple_vectorfield_example(follow_path=True)
    run_single_simple_vectorfield_example(follow_path=True)
