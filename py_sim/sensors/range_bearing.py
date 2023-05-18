"""range_bearing.py Provides functions for calculating the range bearing sensor readings to emulate lidar
"""

import numpy as np
from py_sim.tools.sim_types import RangeBearingMeasurements, UnicyleStateProtocol, TwoDimArray
from py_sim.worlds.polygon_world import PolygonWorld

class RangeBearingSensor:
    """ Class that stores the variables needed for calculating a range bearing measurement and interpreting those
        readings with respect to the vehicle
    """
    def __init__(self, n_lines: int, max_dist: float) -> None:
        """Initializes the readings for the sensors and the interpretation of the readings with respect to the vehicle

            Inputs:
                n_lines: Number of lines in the sensor class
                max_dist: Maximum distance that each sensor can see
        """
        # Store the input variables
        self.n_lines = n_lines
        self.max_dist = max_dist
        self.ind_left: list[int] = [] # Stores indices for sensors on the left of the vehicle
        self.ind_right: list[int] = [] # Stores indices for sensors on the right of the vehicle
        self.ind_front: list[int] = [] # Stores indices for sensors on the front of the vehicle
        self.ind_rear: list[int] = [] # Stores indices for sensors on the rear of the vehicle

        # Calculate the sensor orientations
        delta = np.pi/n_lines   # Calculate an offset so that no line is straight forward
        self.orien = np.linspace(start=delta, stop=2.*np.pi+delta, num=n_lines, endpoint=False)

        # Adjust the sensors and store orientation indices
        for (ind,angle) in enumerate(self.orien):
            # Adjust the sensor angle to be between -pi and pi
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            self.orien[ind] = angle

            # Store the index to be on left/right of the vehicle
            if angle > 0.:
                self.ind_left.append(ind)
            else:
                self.ind_right.append(ind)

            # Store the index to be on the front/rear of the vehicle
            if np.abs(angle) < np.pi/2.:
                self.ind_front.append(ind)
            else:
                self.ind_rear.append(ind)

    def calculate_range_bearing_measurement(self, pose: UnicyleStateProtocol, world: PolygonWorld) -> RangeBearingMeasurements:
        """Calculates the range / bearing measurements given the current pose of the vehicle

            Inputs:
                pose: Position and orientation of the vehicle
        """

        # Initialize the output
        measurement = RangeBearingMeasurements()
        measurement.bearing = (self.orien + pose.psi).tolist()

        # Loop through the bearing and calculate the sensor measurement
        for (ind, angle) in enumerate(measurement.bearing):
            # Update the angle to be between pi and -pi
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            measurement.bearing[ind] = angle

            # Determine the closest intersection point
            edge = np.array([[pose.x, pose.x+np.cos(angle)],
                             [pose.y, pose.y+np.sin(angle)]])
            intersection = world.find_closest_obstacle(edge)

            # Store the resulting data
            if intersection is None:
                measurement.range.append(np.inf)
                measurement.location.append(TwoDimArray())
            else:
                measurement.range.append(intersection[0])
                measurement.location.append(TwoDimArray(vec=intersection[1]))

        return measurement
