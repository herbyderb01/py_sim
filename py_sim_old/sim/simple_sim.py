"""simple_sim.py: Creates a simple simulation for a vehicle and a given controller

Functions:

Classes:
    SimpleManifest: Manifest class defining the dynamics and control to be used

"""
from typing import Callable, Generic, TypeVar, Protocol

from py_sim.sim.generic_sim import Data, SimParameters, Slice
from py_sim.tools.sim_types import Dynamics, DynamicsUpdate, Input, State, StateType, InputType

# Generic definitions
ControlParams = TypeVar("ControlParams") # Used to represent generic control parameters

class SimpleManifest(Protocol[ControlParams, StateType, InputType]):
    """Defines all of the funcitons and scenario specific parameters for the simple simulation
    """

    # def __init__(self) -> None:
    #     super().__init__()

    # The dynamics function to be used
    #   Inputs:
    #       State: The current state of the vehicle
    #       Input: The control input
    #   Outputs:
    #       State: The time derivative of the state being output
    dynamics: Dynamics[StateType, InputType]

    # Dynamics update function type
    #   Inputs:
    #       dynamics: a function handle for calculating the time derivative of the state
    #       initial: the starting state of the vehicle
    #       input: the control input to be applied
    #       dt: the time step of the update
    #   Outputs:
    #       The resulting time derivative of the state
    dynamic_update: DynamicsUpdate[StateType, InputType]

    # Vector field controller
    #   Inputs:
    #       float: The current time
    #       State: The current state of the vehicle
    #       ControlParams: Control specific variables
    #   Outputs:
    #       Input: The control input to be applied to the system
    @staticmethod
    def control(time: float, state: StateType, params: ControlParams) -> InputType: ...
    #control: Callable[[float, StateType, ControlParams], InputType]
    control_params: ControlParams # Control specific variables

class SimpleSim(Generic[ControlParams, StateType, InputType]):
    """Framework for implementing a simulator that just tests out a feedback controller"""
    def __init__(self, manifest: SimpleManifest[ControlParams, StateType, InputType], params: SimParameters[StateType]) -> None:
        """Initialize the simulation

            Inputs:
                initial_state: The initial state of the simulator
                sim_param: The parameters of the simulation

        """
        # Create and store the data
        initial_slice: Slice[StateType] = Slice(state=params.initial_state, time=params.t0)
        self.data: Data[StateType] = Data(current=initial_slice)
        self.params = params
        self.manifest = manifest

    async def setup(self) -> None:
        """Setup all of the storage and plotting"""

    async def update(self) -> None:
        """Calls all of the update functions
            * Gets the latest vector to be followed
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """
        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

        # Calculate the control to follow the vector
        control = self.manifest.control(self.data.current.time,
                                        self.data.current.state,
                                        self.manifest.control_params)

        # Update the state using the latest control
        # self.data.next.state.state = self.manifest.dynamic_update(dynamics=self.manifest.dynamics,
        #                                                     initial=self.data.current.state,
        #                                                     control=control, dt=self.params.sim_step)

        self.data.next.state.state = self.manifest.dynamic_update(self.manifest.dynamics,
                                                        self.data.current.state,
                                                        control, self.params.sim_step)


    async def plot(self) -> None:
        """Plot the current values and state"""

    async def post_process(self) -> None:
        """Process the results"""
        print("Final state: ", self.data.current.state.state)
