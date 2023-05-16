"""vectorfields.py defines various vector fields and associated parameters
"""

from py_sim.tools.sim_types import UnicyleStateProtocol, TwoDimArray
from typing import Protocol
import numpy as np

class VectorField(Protocol):
    """Defines the functions needed for a vector field class"""
    def calculate_vector(self, state: UnicyleStateProtocol, time: float) ->TwoDimArray:
        """Calculates a vector given the time and unicycle state"""


class GoToGoalField:
    """Defines a go-to-goal vector field"""
    def __init__(self, x_g: TwoDimArray, v_max: float, sig: float = 1.) -> None:
        """Initializes the go-to-goal vector field

            Inputs:
                x_g: 2D goal position that vector field attempts to reach
                v_max: Maximum translational velocity
                sig: Convergence factor for going to zero velocity
        """
        self.x_g = x_g          # Goal
        self.v_max = v_max      # Max velocity
        self.sig_sq = sig**2    # Square for convergence

    def calculate_vector(self, state: UnicyleStateProtocol, _: float = 0.) -> TwoDimArray:
        """Calculates a vector from the state to the goal, the vector is scaled to respect max velocity

            Inputs:
                state: State of the vehicle

            Outputs:
                Vector pointing towards the goal
        """
        # Calculate the vector pointing towards the goal
        g = self.x_g.state - state.position

        # Scale the magnitude of the resulting vector
        dist = np.linalg.norm(g)
        v_g = self.v_max * (1.-np.exp(-dist**2/self.sig_sq))

        # Scale the vector
        if dist > 0.:
            g = (v_g/dist)*g
            result = TwoDimArray(vec=g)
        else:
            result = TwoDimArray(x=0., y=0.)
        return result