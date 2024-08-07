import numpy as np

def two_point_five_carrot_projection() -> None:
    """Defines the carrot projection problem"""

    # Initialize problem values
    p1 = np.array([[4],[3]])    # Line segment points
    p2 = np.array([[10], [9]])
    p3 = np.array([[19], [7]])
    q = np.array([[10], [7]])   # Point being projected

    # Calculate distance along line segments
    d_12 = np.linalg.norm(p2-p1)
    print("d_12 = ", d_12)
    d_23 = np.linalg.norm(p3-p2)
    print("d_23 = ", d_23)

    ### Part 1: Projection of point onto the line segments
    # Calculate the unit vectors
    del_1 = p2-p1
    u1 = del_1 / np.linalg.norm(del_1)
    print("u1 = \n", u1)
    del_2 = p3-p2
    u2 = del_2 / np.linalg.norm(del_2)
    print("u2 = \n", u2)

    # Projection onto line segment 1
    s1 = u1.transpose()@(q-p1)
    print("s1 = ", s1)
    q1 = p1 + s1*u1
    print("q1 = \n", q1)

    # Projection onto line segment 2
    s2 = u2.transpose()@(q-p2)
    print("s2 = ", s2)
    q2 = p2 + s2*u2
    print("q2 = \n", q2)

    # Calculate the distances
    d1 = np.linalg.norm(q-q1)
    d2 = np.linalg.norm(q-q2)
    print("d1 = ", d1, ", d2 = ", d2)

    # Calculate the carrot point (5 in front of projection point)
    s_p2 = d_12
    s_23 = 5 - (s_p2 - s1)
    print("s_23 = ", s_23)

    q_c = p2 + s_23*u2
    print("q_c = \n", q_c)




if __name__ == "__main__":
    print("Starting problem 2.5 - Carrot projection")
    two_point_five_carrot_projection()