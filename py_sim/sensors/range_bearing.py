"""range_bearing.py Provides functions for calculating the range bearing sensor readings to emulate lidar
"""

import numpy as np
from py_sim.tools.sim_types import (
    LocationStateType,
    RangeBearingMeasurements,
    TwoDimArray,
    UnicycleStateProtocol,
)
from py_sim.worlds.polygon_world import PolygonWorld


class RangeBearingSensor:
    """ Class that stores the variables needed for calculating a range bearing measurement and interpreting those
        readings with respect to the vehicle

    Attributes:
        n_lines(int): Number of lines in the sensor class
        max_dist(float): Maximum distance that each sensor can see
        ind_left(list[int]): Stores indices for sensors on the left of the vehicle
        ind_right(list[int]): Stores indices for sensors on the right of the vehicle
        ind_front(list[int]): Stores indices for sensors on the front of the vehicle
        ind_rear(list[int]): Stores indices for sensors on the rear of the vehicle
        orien(NDArray[Any]): Array of orientations for each of the sensors on the body
        delta: The angle difference between two lines of measurement
    """
    def __init__(self, n_lines: int, max_dist: float) -> None:
        """Initializes the readings for the sensors and the interpretation of the readings with respect to the vehicle

        Args:
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
        self.delta = 2*np.pi/n_lines   # Calculate an offset so that no line is straight forward
        self.orien = np.linspace(start=self.delta/2., stop=2.*np.pi+self.delta/2., num=n_lines, endpoint=False)

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

    def calculate_range_bearing_measurement(self, pose: LocationStateType, world: PolygonWorld) -> RangeBearingMeasurements:
        """Calculates the range / bearing measurements given the current pose of the vehicle

            Args:
                pose: Position. If also has orientation (UnicycleStateProtocol) then orientation used to calculate the bearing
                world: Polygon world in which the vehicle is operating

            Returns:
                RangeBearingMeasurements: Measurements given the vehicle pose and polygon world
        """

        # Initialize the output
        measurement = RangeBearingMeasurements()
        if isinstance(pose, UnicycleStateProtocol):
            measurement.bearing = (self.orien + pose.psi).tolist()
        else:
            measurement.bearing = self.orien.tolist()

        # Loop through the bearing and calculate the sensor measurement
        for (ind, angle) in enumerate(measurement.bearing):
            # Update the angle to be between pi and -pi
            angle = np.arctan2(np.sin(angle), np.cos(angle))
            measurement.bearing[ind] = angle

            # Determine the closest intersection point
            edge = np.array([[pose.x, pose.x+self.max_dist*np.cos(angle)],
                             [pose.y, pose.y+self.max_dist*np.sin(angle)]])
            intersection = world.find_closest_obstacle(edge)

            # Store the resulting data
            if intersection is None:
                measurement.range.append(np.inf)
                measurement.location.append(TwoDimArray(x=pose.x+self.max_dist*np.cos(angle),
                                                        y=pose.y+self.max_dist*np.sin(angle)))
            else:
                measurement.range.append(intersection[0])
                measurement.location.append(TwoDimArray(vec=intersection[1]))

        return measurement

    def create_measurement_from_range(self, pose: LocationStateType, ranges: list[float]) -> RangeBearingMeasurements:
        """Calculate the measurement given the position and ranges
        """
        # Initialize the output
        measurement = RangeBearingMeasurements()
        if isinstance(pose, UnicycleStateProtocol):
            measurement.bearing = (self.orien + pose.psi).tolist()
        else:
            measurement.bearing = self.orien.tolist()

        # Initialize ranges to the infinite distance measurement
        q = pose.position
        for bearing in measurement.bearing:
            # Calculate the position of the max distance sensor reading
            loc = q + self.max_dist* np.array([[np.cos(bearing)],
                                               [np.sin(bearing)]])

            measurement.range.append(np.inf)
            measurement.location.append(TwoDimArray(vec=loc))

        # Ensure that the ranges match the bearing, if not then an infinite value will be used for each range
        if len(ranges) != self.n_lines:
            print("Warning: incorrect range number, solely storing inf")
            return measurement

        # Loop through and calculate the non-infinite sensor measurments
        for k, range_k in enumerate(ranges):
            if range_k < np.inf:
                bearing = measurement.bearing[k]
                loc = q + range_k* np.array([[np.cos(bearing)],
                                           [np.sin(bearing)]])
                measurement.range[k] = range_k
                measurement.location[k] = TwoDimArray(vec=loc)

        return measurement
