"""simple_sim.py: Creates scenarios for launching the simple simulator

Functions:

Classes:

"""

import asyncio

import numpy as np
from py_sim.Dynamics.unicycle import arc_control, UnicycleState
from py_sim.Dynamics.unicycle import dynamics as simple_unicycle
from py_sim.sim.generic_sim import SimParameters, euler_update, run_sim_simple
from py_sim.sim.simple_sim import SimpleSim, SimpleManifest
from py_sim.tools.sim_types import ArcParams

class UnicycleManifest(SimpleManifest[ArcParams]):
    """Creates a manifest that implements the SimpleManifest for the Unicycle
    """
    def __init__(self) -> None:
        super().__init__()
        self.dynamics = simple_unicycle
        self.dynamic_update = euler_update
        self.control_params = ArcParams(v_d=1., w_d=1.)
        self.control = arc_control

def simple_unicycle_sim() -> None:
    """Runs a simple simulation of the unicycle with a circular control law"""
    # Create the simulator
    initial_state = UnicycleState(x = 0., y=0., psi=0.)
    params=SimParameters(initial_state=initial_state)
    sim = SimpleSim(manifest=UnicycleManifest(), params=params)
    asyncio.run(run_sim_simple(sim))


if __name__ == "__main__":
    simple_unicycle_sim()
