import numpy as np
import numpy.typing as npt
from typing import Any

def carrot_projection() -> None:
    """Computes the carrot projection problem in 1.2 of homework 4"""
    v3 = np.array([[19],[7]])
    v2 = np.array([[13],[3]])

    diff = v3-v2
    u = diff / (np.linalg.norm(diff))
    u_hand = 1./np.sqrt(13)*np.array([[3],[2]])

    print("diff = \n", diff)
    print("u=\n", u)
    print("u_hand=\n", u_hand)

def calculate_fillet(x1: npt.NDArray[Any], x2: npt.NDArray[Any], x3: npt.NDArray[Any], r: float) -> None:
    """Calculates the fillet values

    Args:
        x1: first point on path
        x2: second point on path
        x3: third point on path
        r: radius of curvature
    """

    # Calculate unit vectors pointing along the line segment
    u12 = (x2-x1) / np.linalg.norm(x2-x1)
    u23 = (x3-x2) / np.linalg.norm(x3-x2)

    # Calculate the angle between the two vectors
    J = np.array([[0, -1],
                  [1, 0]])
    h12 = J@u12 # Vector pointing to the left from u12
    zeta = np.sign(u23.transpose()@J@u12)
    gamma = zeta * np.arccos(u23.transpose()@u12)

    # Calculate the distance between the two vectors
    d = np.abs(r*(1-np.cos(gamma))/np.sin(gamma))
    print("Distance from center to waypoint: ", d)

    # Calculate the center of the arc
    x_s = x2 - u12*d # Starting point of the arc
    x_c = x_s + r*zeta*h12

    # Calculate the end-point
    x_e = x2 + u23*d

    # Print out results
    print("center (c) = \n", x_c)
    print("\ndistance (d) = \n", d)
    print("\nx_s = \n", x_s)
    print("\nx_e = \n", x_e)


def main() -> None:
    """Caluculates the arc vales for two transitions"""

    # Create the points of the line
    v1 = np.array([[4],[3]])
    v2 = np.array([[13],[3]])
    v3 = np.array([[19],[7]])
    v4 = np.array([[17],[3]])

    # Calculate the fillet curve for the first transition
    print("First transition")
    calculate_fillet(x1=v1, x2=v2, x3=v3, r=2)

    # Calculate the fillet curve for the second transition
    print("\n\n\nFirst transition")
    calculate_fillet(x1=v2, x2=v3, x3=v4, r=2)

if __name__ == "__main__":
    main()
    #carrot_projection()
