"""vectorfields.py defines various vector fields and associated parameters
"""

import numpy as np
from py_sim.tools.sim_types import TwoDArrayType, TwoDimArray, VectorField


class GoToGoalField:
    """Defines a go-to-goal vector field

    Attributes:
        x_g(TwoDimArray): 2D goal position that vector field attempts to reach
        v_max(float): Maximum translational velocity
        sig_sq(float): Convergence factor for going to zero velocity, squared
    """
    def __init__(self, x_g: TwoDimArray, v_max: float, sig: float = 1.) -> None:
        """Initializes the go-to-goal vector field

        Args:
            x_g: 2D goal position that vector field attempts to reach
            v_max: Maximum translational velocity
            sig: Convergence factor for going to zero velocity
        """
        self.x_g = x_g          # Goal
        self.v_max = v_max      # Max velocity
        self.sig_sq = sig**2    # Square for convergence

    def calculate_vector(self, state: TwoDArrayType, time: float = 0.) -> TwoDimArray: # pylint: disable=unused-argument
        """Calculates a vector from the state to the goal, the vector is scaled to respect max velocity

        Args:
            state: State of the vehicle
            time: Time of the state

        Returns:
            TwoDimArray: Vector pointing towards the goal
        """
        print("Add scaling function!!!")
        g = self.x_g.position - state.position
        return TwoDimArray(vec=g)

class AvoidObstacle:
    """Defines an avoid obstacle vector field with a finite sphere of influence

    Attributes:
        x_o(TwoDimArray): Position of the obstacle to avoid
        v_max(float): Maximum velocity of the vector field
        S(float): Sphere of influence
        R(float): Radius of max effect
    """
    def __init__(self, x_o: TwoDimArray, v_max: float, S: float = 5., R: float = 1.) -> None:
        """Create the vector field selfect

        Args:
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

    def calculate_vector(self, state: TwoDArrayType, time: float = 0.) -> TwoDimArray: # pylint: disable=unused-argument
        """Calculates a vector from the state to the goal, the vector is scaled to respect max velocity

        Args:
            state: State of the vehicle
            time: Time of the state

        Returns:
            TwoDimArray: Vector pointing towards the goal
        """
        print("Fix me!!!")
        return TwoDimArray()

class SummedField:
    """Defines a vector field that is the summation of passed in fields

    Attributes:
        fields(list[VectorField]): List of individual vector fields to be summed together
        weights(list[float]): Weight associated with each vector field
        v_max(float): maximum allowable velocity
    """
    def __init__(self, fields: list[VectorField], weights: list[float], v_max: float) -> None:
        """Stores a sum of vector fields to be plotted

        Args:
            fields: List of individual vector fields to be summed together
            weights: Weight associated with each vector field
            v_max: maximum allowable velocity
        """
        # Store the data
        self.fields = fields
        self.weights = weights
        self.v_max = v_max

        # Ensure that fields and weights have the same number of elements
        if len(fields) != len(weights):
            raise ValueError("Fields and weights must have the same number of objects")

    def calculate_vector(self, state: TwoDArrayType, time: float = 0.) -> TwoDimArray:
        """Calculates a summed vector and thresholds it to v_max

        Args:
            state: State of the vehicle
            time: Time of the state

        Returns:
            TwoDimArray: Resulting summed vector
        """
        # Get the summed vector
        g = TwoDimArray(x=0., y=0.)
        for (field, weight) in zip(self.fields, self.weights):
            g.state = g.state + weight*field.calculate_vector(state=state, time=time).state

        # Saturate the field to have a maximum velocity of v_max
        print("Fix me!!!")

        return g

class G2GAvoid:
    """Defines a vector field that sums go-to-goal and obstacle fields with update functions for each

    Attributes:
        n_obs(int): The number of obstacles
        x_g(TwoDimArray): The goal position
        v_max(float): Maximum translational velocity
        _go_to_goal(GoToGoalField): The go to goal vector field
        _avoid(list[AvoidObstacle]): The avoid obstacle fields
        _summed(Summedfield): The combined field
    """
    def __init__(self,
                 x_g: TwoDimArray,
                 n_obs: int,
                 v_max: float,
                 S: float,
                 R: float,
                 sig: float = 1.,
                 weight_g2g: float = 1.,
                 weight_avoid: float = 1.) -> None:
        """Initialize the vector field that balances go-to-goal and avoidance

        Args:
            x_g: Goal location
            n_obs: Number of obstacles to avoid
            v_max: Maximum allowable velocity
            S: Sphere of influence of the obstacle avoidance
            R: Radius of max influence for the obstacle avoidance
            sig: The convergence factor for the go-to-goal
            weight_g2g: The weight on the go-to-goal field
            weight_avoid: The weight on the avoidance field
        """
        # Store inputs
        self.n_obs = n_obs

        # Create the go-to-goal vector field
        self._go_to_goal = GoToGoalField(x_g=x_g, v_max=v_max, sig=sig)
        weights: list[float] = [weight_g2g]

        # Create the obstacle avoid vector fields
        self._avoid: list[AvoidObstacle] = []
        for _ in range(n_obs):
            self._avoid.append(AvoidObstacle(x_o=TwoDimArray(x=1.e10, y=1.e10), v_max=v_max, S=S, R=R))
            weights.append(weight_avoid)

        # Create the summed vector field
        fields: list[VectorField] = self._avoid + [self._go_to_goal]
        self._summed = SummedField(weights=weights, fields=fields, v_max=v_max)

    @property
    def x_g(self) -> TwoDimArray:
        """Goal position getter"""
        return self._go_to_goal.x_g

    @x_g.setter
    def x_g(self, val: TwoDimArray) -> None:
        """Set the goal position setter"""
        self._go_to_goal.x_g = val

    @property
    def v_max(self) -> float:
        """The maximum velocity getter"""
        return self._go_to_goal.v_max

    @v_max.setter
    def v_max(self, val: float) -> None:
        """The maximum velocity setter"""
        self._go_to_goal.v_max = val
        self._summed.v_max = val
        for avoid in self._avoid:
            avoid.v_max = val

    def update_obstacles(self, locations: list[TwoDimArray]) -> None:
        """Updates the positions of the obstacles

        Args:
            locations: The locations of the obstacles. Must be of length n_obs
        """
        # Check to ensure the locations are the right size
        if len(locations) != self.n_obs:
            raise ValueError("Cannot pass in a list of locations of the wrong size")

        # Update the locations
        for (avoid, location) in zip(self._avoid, locations):
            avoid.x_o = location

    def calculate_vector(self, state: TwoDArrayType, time: float = 0.) -> TwoDimArray:
        """Calculates a vector from the state to the goal, the vector is scaled to respect max velocity

        Args:
            state: State of the vehicle
            time: Time of the state

        Returns:
            TwoDimArray: Vector pointing towards the goal
        """
        return self._summed.calculate_vector(state=state, time=time)
