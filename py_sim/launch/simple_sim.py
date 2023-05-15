"""simple_sim.py performs a test with a single vehicle"""

import matplotlib.pyplot as plt
from py_sim.dynamics.unicycle import arc_control
from py_sim.dynamics.unicycle import dynamics as unicycle_dynamics
from py_sim.sim.generic_sim import SingleAgentSim, start_simple_sim
from py_sim.sim.integration import euler_update
from py_sim.tools.plotting import PosePlot, PositionPlot, StateTrajPlot
from py_sim.tools.sim_types import ArcParams, UnicycleState, InputType, Dynamics, UnicycleControl, ControlParamType, Control
from typing import TypeVar, Generic


StateType = TypeVar("StateType", bound=UnicycleState)

class SimpleSim(Generic[StateType, InputType, ControlParamType], SingleAgentSim[StateType]):
    """Framework for implementing a simulator that just tests out a feedback controller"""
    def __init__(self,
                dynamics: Dynamics[StateType, InputType],
                controller: Control[StateType, InputType, ControlParamType],
                control_params: ControlParamType
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage"""
        super().__init__()

        # Initialize sim-specific parameters
        self.dynamics: Dynamics[StateType, InputType] = dynamics
        self.controller: Control[StateType, InputType, ControlParamType] = controller
        self.control_params: ControlParamType = control_params

        # Initialize the plotting
        self.fig, self.ax = plt.subplots()
        self.ax.set_title("Vehicle plot")
        self.ax.set_ylim(ymin=-2., ymax=2.)
        self.ax.set_xlim(xmin=-2., xmax=2.)
        self.ax.set_aspect('equal', 'box')

        # Create the desired state plots
        self.state_plots.append(PositionPlot(ax=self.ax, label="Vehicle", color=(0.2, 0.36, 0.78, 1.0)) )
        self.state_plots.append(PosePlot(ax=self.ax, rad=0.2))
        self.state_plots.append(StateTrajPlot(ax=self.ax, label="Vehicle Trajectory", \
                                color=(0.2, 0.36, 0.78, 1.0), location=self.data.current.state))


    def update(self) -> None:
        """Calls all of the update functions
            * Gets the latest vector to be followed
            * Calculate the control to be executed
            * Update the state
            * Update the time
        """

        # Update the time by sim_step
        self.data.next.time = self.data.current.time + self.params.sim_step

        # Calculate the control to follow a circle
        control:InputType = self.controller(time=self.data.current.time,
                                state=self.data.current.state, # type: ignore
                                params=self.control_params)

        # Update the state using the latest control
        self.data.next.state.state = euler_update(  dynamics=self.dynamics,
                                                    control=control,
                                                    initial=self.data.current.state,
                                                    dt=self.params.sim_step) # type: ignore

if __name__ == "__main__":
    # Runs the simulation and the plotting
    control_params = ArcParams(v_d=1., w_d= 1.)
    sim = SimpleSim(dynamics=unicycle_dynamics, controller=arc_control, control_params=control_params)
    start_simple_sim(sim=sim)
