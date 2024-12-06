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
        # print("Add scaling function!!!")
        g = self.x_g.position - state.position

        mag_g = np.linalg.norm(g)
        if mag_g > self.v_max:
            g = g/(mag_g*self.v_max)
        else:
            g = g

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
        # Extract the x and y coordinates from the state and obstacle position
        obstacle_to_state_x = state.x - self.x_o.x
        obstacle_to_state_y = state.y - self.x_o.y
        
        # Create a vector representing the direction from the obstacle to the agent
        obstacle_to_state = np.array([obstacle_to_state_x, obstacle_to_state_y])
        
        # Calculate the distance from the agent to the obstacle
        distance = np.linalg.norm(obstacle_to_state)

        # If distance is greater than the sphere of influence, return a zero vector
        if distance > self.S:
            return TwoDimArray(0, 0)

        # Avoid division by zero if agent is exactly at the obstacle's position
        if distance == 0:
            return TwoDimArray(0, 0)

        # Calculate the scaling factor
        scale = self.v_max
        if distance <= self.R:
            scale = self.v_max  # Within radius of max effect
        else:
            # Scale velocity based on how close it is to the sphere of influence
            scale *= float(self.S - distance) / float(self.S - self.R)

        # Normalize the vector and scale it
        avoidance_vector = (obstacle_to_state / distance) * scale

        # Return the result as a TwoDimArray (assuming vec can accept an array)
        return TwoDimArray(vec=avoidance_vector)


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
        """Calculates a summed vector and thresholds it to v_max.

        Args:
            state: State of the vehicle
            time: Time of the state

        Returns:
            TwoDimArray: Resulting summed vector, capped at v_max
        """
        # Initialize the summed vector to zero
        g = TwoDimArray(x=0., y=0.)

        # Sum the weighted vectors from all fields
        for (field, weight) in zip(self.fields, self.weights):
            field_vector = field.calculate_vector(state=state, time=time)
            g.x += weight * field_vector.x
            g.y += weight * field_vector.y

        # Calculate the magnitude of the resulting vector
        magnitude = np.sqrt(g.x**2 + g.y**2)

        # Check if the magnitude exceeds v_max
        if magnitude > self.v_max:
            # Normalize the vector and scale it to v_max
            scale_factor = self.v_max / magnitude
            g.x *= scale_factor
            g.y *= scale_factor

        # Return the resulting vector, possibly scaled down to v_max
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
