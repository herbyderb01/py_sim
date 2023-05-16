"""simple_sim.py performs a test with a single vehicle"""

from typing import TypeVar

import matplotlib.pyplot as plt
from py_sim.dynamics.unicycle import arc_control
from py_sim.dynamics.unicycle import dynamics as unicycle_dynamics
from py_sim.sim.generic_sim import SingleAgentSim, start_simple_sim
from py_sim.tools.plotting import (
    PosePlot,
    PositionPlot,
    StateTrajPlot,
    UnicycleTimeSeriesPlot,
)
from py_sim.tools.sim_types import (
    ArcParams,
    Control,
    ControlParamType,
    Dynamics,
    InputType,
    UnicycleControl,
    UnicycleState,
)

# Limit the state type to be a unicycle state
StateType = TypeVar("StateType", bound=UnicycleState)

class SimpleSim(SingleAgentSim[StateType, InputType, ControlParamType]):
    """Framework for implementing a simulator that just tests out a feedback controller"""
    def __init__(self,
                initial_state: StateType,
                dynamics: Dynamics[StateType, InputType],
                controller: Control[StateType, InputType, ControlParamType],
                control_params: ControlParamType,
                input_example: InputType
                ) -> None:
        """Creates a SingleAgentSim and then sets up the plotting and storage"""
        super().__init__(initial_state=initial_state,
                         dynamics=dynamics,
                         controller=controller,
                         control_params=control_params,
                         input_example=input_example)

        # Initialize the plotting of the vehicle visualization
        fig, ax = plt.subplots()
        self.figs.append(fig)
        self.axes['Vehicle_axis'] = ax
        ax.set_title("Vehicle plot")
        ax.set_ylim(ymin=-2., ymax=2.)
        ax.set_xlim(xmin=-2., xmax=2.)
        ax.set_aspect('equal', 'box')

        # Create the desired state plots
        self.state_plots.append(PositionPlot(ax=ax, label="Vehicle", color=(0.2, 0.36, 0.78, 1.0)) )
        self.state_plots.append(PosePlot(ax=ax, rad=0.2))

        # Create the state trajectory plot
        self.data_plots.append(StateTrajPlot(ax=ax, label="Vehicle Trajectory", \
                                color=(0.2, 0.36, 0.78, 1.0), location=self.data.current.state))

        # Create the desired data plots
        state_plot = UnicycleTimeSeriesPlot[StateType](color=(0.2, 0.36, 0.78, 1.0))
        self.data_plots.append(state_plot)
        self.figs.append(state_plot.fig)



if __name__ == "__main__":
    # Runs the simulation and the plotting
    arc_params = ArcParams(v_d=1., w_d= 1.)
    state_initial = UnicycleState(x = 0., y= 0., psi= 0.)
    sim = SimpleSim(initial_state=state_initial,
                    dynamics=unicycle_dynamics,
                    controller=arc_control,
                    control_params=arc_params,
                    input_example=UnicycleControl())
    sim.params.sim_plot_period = 0.2
    sim.params.sim_update_period = 0.01
    start_simple_sim(sim=sim)
