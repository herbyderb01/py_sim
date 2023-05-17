"""vectorfields.py defines various vector fields and associated parameters
"""

import numpy as np
from py_sim.tools.sim_types import TwoDimArray, UnicyleStateProtocol


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

    def calculate_vector(self, state: UnicyleStateProtocol, time: float = 0.) -> TwoDimArray: # pylint: disable=unused-argument
        """Calculates a vector from the state to the goal, the vector is scaled to respect max velocity

            Inputs:
                state: State of the vehicle
                time: Time of the state

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

class AvoidObstacle:
    """Defines an avoid obstacle vector field with a finite sphere of influence"""
    def __init__(self, x_o: TwoDimArray, v_max: float, S: float = 5., R: float = 1.) -> None:
        """Create the vector field selfect

            Inputs:
                x_o: Position of the obstacle to avoid
                v_max: Maximum velocity of the vector field
                S: Sphere of influence
                R: Radius of max effect
        """
        # Store inputs
        self.x_o = x_o
        self.v_max = v_max
        self.S = S
        self.R = R

    def calculate_vector(self, state: UnicyleStateProtocol, time: float = 0.) -> TwoDimArray: # pylint: disable=unused-argument
        """Calculates a vector from the state to the goal, the vector is scaled to respect max velocity

            Inputs:
                state: State of the vehicle
                time: Time of the state

            Outputs:
                Vector pointing towards the goal
        """
        g = state.position - self.x_o.state

        # Scale the magnitude of the resulting vector
        dist = float(np.linalg.norm(g))
        scale = 1.
        if dist > self.S:
            scale = 0
        elif dist > self.R:
            scale = (self.S - dist) / (self.S - self.R)
        v_g = self.v_max * scale # Scaled desired velocity

        # Output g
        if dist > 0: # Avoid dividing by zero
            g = v_g/dist * g # Dividing by dist is dividing by the norm
        else: # Choose a random position if you are on top of the obstacle
            g = np.random.random(2)

        return TwoDimArray(vec=g)
