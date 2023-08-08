"""test_projections.py tests out various functions in the tools/projections.py module"""

from py_sim.tools.projections import project_point_to_edge, project_point_to_line, carrot_point, calculate_line_distances
import numpy as np
import numpy.typing as npt
from typing import Any
import matplotlib.pyplot as plt
import time


def plot_projection_error(edge: npt.NDArray[Any],
                          point: npt.NDArray[Any],
                          proj: npt.NDArray[Any],
                          closer: npt.NDArray[Any] ) -> None:
    """Plots the projection error

    Args:
        edge: The edge to which the projection is made
        point: The point being projected
        proj: The calculated projection of the point onto the line
        closer: The point that was found to be closer to the point
    """
    _, ax = plt.subplots()

    ax.plot(edge[0,:], edge[1,:], 'b')
    ax.plot(point[0], point[1], 'go')
    ax.plot(proj[0], proj[1], 'go')
    ax.plot(closer[0], closer[1], 'ro')

    ax.set_aspect('equal', 'box')
    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    print("Red point should be further from point than the green point")
    print("Close plot to error out")
    plt.show()


def test_project_point_to_edge():
    """Tests the project_point_to_edge function

    Two tests are performed:
        1. Test that the point found is closest to the point given than any other point on the line
        2. Test that the line segment found by the point found is orthogonal to the input line segment (if point found is not the edge case)
    """

    # Tests 1 and 2: Radomly generate points and test for closeness and orthogonoality
    for k in range(100):
        print("test_project_point_to_edge test series ", k)

        # Create a random edge
        edge = (np.random.random(size=(2,2))-0.5)*10.
        a = edge[:, [0]]
        b = edge[:, [1]]
        edge_length = float(np.linalg.norm( b-a ) )
        u = (b-a) / edge_length # Unit vector

        for _ in range(100):
            # Create s_min and s_max values
            if k < 50:
                s_min = 0.
                s_max = np.inf
            else:
                s_min = np.random.random()*edge_length
                s_max = np.random.random()*(edge_length-s_min) + s_min

            # Create a random point
            point = (np.random.random(size=(2,1))-0.5)*10

            # Calculate the projection
            (proj, s) = project_point_to_edge(edge=edge,
                                              point=point,
                                              s_min=s_min,
                                              s_max=s_max)
            dist = np.linalg.norm(proj-point)

            # Test 1: check to see if closer than any other point
            bound = np.min([s_max, edge_length])
            for ds in  np.arange(s_min, bound, 0.01):
                c = a + ds*u
                if np.linalg.norm(c-point) < dist:
                    print("a = ", a, ", b = ", b, ", proj = ", proj, ", closer point = ", c)
                    print("ds = ", ds, ", s_min = ", s_min, ", s_max = ", s_max)
                    plot_projection_error(edge=edge,
                                          point=point,
                                          proj=proj,
                                          closer=c)
                    raise ValueError("Point c is closer to the original point than proj is")

            # Test 2: check orthogonality
            if s > s_min and s < bound:
                product = u.transpose()@(proj-point)
                if abs(product.item(0)) > 1.e-6:
                    print("a = ", a, ", b = ", b, ", proj = ", proj, ", dot product = ", product)
                    raise ValueError("Dot product is not zero")

def test_project_point_to_line():
    """Tests the project point to line function

    One test is performed:
        1. All points along the line are sampled and verified to be further than the projection
    """

    # Create various lines
    for k in range(10):
        print("test_project_point_to_line: test series ", k)
        line = (np.random.random(size=(2,k+2))-.5)*10.
        s_vals = calculate_line_distances(line=line)

        # Create various points for testing
        for p in range(100):
            # Create a random point
            point = (np.random.random(size=(2,1))-0.5)*10

            # Determine the range for lookup
            if p < 50:
                s_min = 0.
                s_max = np.inf
            else:
                s_min = np.random.random()*s_vals[-1]
                s_max = np.random.random()*(s_vals[-1]-s_min) + s_min

            # Create the projection
            (proj, s_val, k_prev) = project_point_to_line(point=point,
                                                          line=line,
                                                          s_vals=s_vals,
                                                          s_min=s_min,
                                                          s_max=s_max)

            # Test 1: check distance with points along the line
            for s in np.arange(s_min, np.min([s_max, s_vals[-1]]), .01):
                carrot = carrot_point(line=line,
                                      s_vals=s_vals,
                                      s_des=s)
                if np.linalg.norm(carrot-point) < np.linalg.norm(proj-point):
                    print("line = ", line, "\npoint = ", point, "\nproj = ", proj, "\ncarrot = ", carrot)
                    raise ValueError("Carrot point is closer to the point then the projected value")





if __name__ == "__main__":
    test_project_point_to_edge()
    test_project_point_to_line()
    print("Completed test")
