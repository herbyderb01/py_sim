"""simple_sim.py: Creates scenarios for launching the simple simulator

Functions:

Classes:

"""

import asyncio

import numpy as np
from py_sim.Dynamics.unicycle import arc_control, UnicycleState, UnicycleInput
from py_sim.Dynamics.unicycle import dynamics as simple_unicycle
from py_sim.sim.generic_sim import SimParameters, euler_update, run_sim_simple
from py_sim.sim.simple_sim import SimpleSim, SimpleManifest
from py_sim.tools.sim_types import ArcParams, Dynamics, DynamicsUpdate
from typing import Callable

class UnicycleManifest():
    """Creates a manifest that implements the SimpleManifest for the Unicycle
    """
    dynamics: Dynamics[UnicycleState, UnicycleInput] = simple_unicycle
    dynamic_update: DynamicsUpdate[UnicycleState, UnicycleInput] = euler_update
    control_params = ArcParams(v_d=1., w_d=1.)
    # control: Callable[[float, UnicycleState, ArcParams],UnicycleInput] = staticmethod(arc_control)
    # control = staticmethod(arc_control)
    @staticmethod
    def control(time: float, state: UnicycleState, params: ArcParams) -> UnicycleInput:
        return arc_control(time, state, params)

def simple_unicycle_sim() -> None:
    """Runs a simple simulation of the unicycle with a circular control law"""
    # Create the simulator
    initial_state = UnicycleState(x = 0., y=0., psi=0.)
    params=SimParameters(initial_state=initial_state)
    manifest: SimpleManifest[ArcParams, UnicycleState, UnicycleInput] = UnicycleManifest()
    sim = SimpleSim(manifest=manifest, params=params)
    asyncio.run(run_sim_simple(sim))


if __name__ == "__main__":
    simple_unicycle_sim()
