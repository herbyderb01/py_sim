"""vector_field.py: Define classes to manage the lifecycle

Functions:

Classes:

"""
from typing import Callable, Generic, TypeVar

from py_sim.sim.generic_sim import Data, SimParameters, Slice
from py_sim.tools.sim_types import Dynamics, DynamicsUpdate, Input, State, TwoDimVector

# Generic definitions
VectorParams = TypeVar("VectorParams") # Used to represent generic vector parameters
ControlParams = TypeVar("ControlParams") # Used to represent generic control parameters

class VectorFollowManifest(Generic[VectorParams, ControlParams]):
    """Defines all of the functions and scenario specific parameters for the
       vectorfield simulation
    """
    def __init__(self) -> None:
        super().__init__()

        # The dynamics function to be used
        #   Inputs:
        #       State: The current state of the vehicle
        #       Input: The control input
        #   Outputs:
        #       State: The time derivative of the state being output
        self.dynamics: Dynamics

        # Dynamics update function type
        #   Inputs:
        #       dynamics: a function handle for calculating the time derivative of the state
        #       initial: the starting state of the vehicle
        #       input: the control input to be applied
        #       dt: the time step of the update
        #   Outputs:
        #       The resulting time derivative of the state
        self.dynamic_update: DynamicsUpdate

        # Vector field controller
        #   Inputs:
        #       float: The current time
        #       State: The current state of the vehicle
        #       TwoDimVector: The vector to be followed
        #       ControlParams: Control specific variables
        #   Outputs:
        #       Input: The control input to be applied to the system
        self.control: Callable[[float, State, TwoDimVector, ControlParams], Input]
        self.control_params: ControlParams # Control specific variables


        # The vector field parameters and function
        #   Inputs:
        #       float: The current time
        #       State: The current state of the vehicle
        #       VectorParams: Control specific variables
        #   Outputs:
        #       TwoDimVector: The vector to be followed
        self.field: Callable[[float, State, VectorParams], TwoDimVector] # Function to produce vector
        self.vec_params: VectorParams # Constant parameters defining the vector field

        # Initial state of the vehicle at beginning of the simulation
        self.initial_state: State

        # Simulation parameters for timing
        self.sim_params: SimParameters

# class VectorFollowSim():
#     """Generic sim for testing out vector following scenarios"""

#     def __init__(self, manifest: VectorFollowManifest) -> None:
#         """Initialize the simulation

#             Inputs:
#                 initial_state: The initial state of the simulator
#                 sim_param: The parameters of the simulation

#         """
#         # Create and store the data
#         initial_slice = Slice(state=manifest.initial_state, time=manifest.sim_params.t0)
#         self.data = Data(current=initial_slice)
#         self.params = manifest.sim_params
#         self.manifest = manifest

#     async def setup(self) -> None:
#         """Setup all of the storage and plotting"""

#     async def update(self) -> None:
#         """Calls all of the update functions
#             * Gets the latest vector to be followed
#             * Calculate the control to be executed
#             * Update the state
#             * Update the time
#         """
#         # Update the time by sim_step
#         self.data.next.time = self.data.current.time + self.params.sim_step

#         # Calculate the vector to be followed
#         vec = self.manifest.field(self.data.current.time, self.data.current.state, self.manifest.vec_params)

#         # Calculate the control to follow the vector
#         control = self.manifest.control(self.data.current.time, self.data.current.state, vec,
#                                         self.manifest.control_params)

#         # # Update the state using the latest control
#         # self.data.next.state = self.manifest.dynamic_update(dynamics=self.manifest.dynamics,
#         #                                                     initial=self.data.current.state,
#         #                                                     input=control, dt=self.params.sim_step)


#     async def plot(self) -> None:
#         """Plot the current values and state"""

#     async def post_process(self) -> None:
#         """Process the results"""
