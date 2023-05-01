"""simple_sim.py: Creates scenarios for launching the simple simulator

Functions:

Classes:

"""

import asyncio

import numpy as np
from py_sim.Dynamics.unicycle import arc_control
from py_sim.Dynamics.unicycle import dynamics as simple_unicycle
from py_sim.sim.generic_sim import SimParameters, euler_update, run_sim_simple
from py_sim.sim.simple_sim import SimpleSim
from py_sim.tools.sim_types import ArcParams


class UnicycleManifest():
    """Creates a manifest that implements the SimpleManifest for the Unicycle
    """
    def __init__(self) -> None:
        self.dynamics = simple_unicycle
        self.dynamic_update = euler_update
        self.control_params = ArcParams(v_d=1., w_d=1.)
        self.control = arc_control

def simple_unicycle_sim():
    """Runs a simple simulation of the unicycle with a circular control law"""
    params=SimParameters(initial_state=np.array([[0.], [0.], [0.]]))
    sim = SimpleSim(manifest=UnicycleManifest(), params=params)
    asyncio.run(run_sim_simple(sim))


if __name__ == "__main__":
    simple_unicycle_sim()
