import numpy as np
import numpy.typing as npt
from typing import Any

def carrot_projection() -> None:
    """Computes the carrot projection problem in 1.2 of homework 4"""
    v3 = np.array([[19],[7]])
    v2 = np.array([[13],[3]])
    v1 = np.array([[5], [2]])

    # Compute projection
    q = np.array([[12],[1]])
    u_12 = (v2-v1)/np.linalg.norm(v2-v1)
    q_1q = (q-v1)
    s = q_1q.transpose()@u_12
    p = v1 + u_12*s
    print("u_12: \n", u_12)
    print("Projection: \n", p)
    s_p2 = np.linalg.norm(v2-p)

    # Carrot point calculation
    diff = v3-v2
    u = diff / (np.linalg.norm(diff))
    print("u_23: \n", u)
    s_12 = 5 - s_p2
    c = v2 + u*s_12

    print("s_12 = ", s_12)
    print("carrot=\n", c)

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


def fillet() -> None:
    """Caluculates the arc vales for two transitions"""

    # Create the points of the line
    v1 = np.array([[4],[3]])
    v2 = np.array([[13],[3]])
    v3 = np.array([[19],[7]])
    v4 = np.array([[24],[5]])

    # # Calculate the fillet curve for the first transition
    # print("First transition")
    # calculate_fillet(x1=v1, x2=v2, x3=v3, r=2)

    # Calculate the fillet curve for the second transition
    print("\n\n\nFirst transition")
    calculate_fillet(x1=v2, x2=v3, x3=v4, r=2)

if __name__ == "__main__":

    #carrot_projection()
    fillet()
