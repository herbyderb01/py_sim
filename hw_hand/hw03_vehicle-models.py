import numpy as np

def one_point_five_projection() -> None:
    """Defines the projection problem"""

    # Initialize problem values
    a = np.array([[2],[7]])    # Line segment points
    b = np.array([[18], [9]])
    q = np.array([[13], [2]])   # Point being projected

    # Calculate the unit vectors
    del_1 = b-a
    u1 = del_1 / np.linalg.norm(del_1)
    print("u1 = \n", u1)

    # Projection onto line segment
    s1 = u1.transpose()@(q-a)
    print("s1 = ", s1)
    q1 = a + s1*u1
    print("q1 = \n", q1)

def two_point_four_velocity_control():
    """Problem 2.4 - Velocity control for simple models"""
    ### Differential drive
    # Define problem parameters
    r = 0.025 # Wheel radius
    L = 0.1 # Length between wheels
    v_d = 3. # Desired translational velocity
    w_d = 2./3. # Desired rotational velocity

    # Solve for the desired inputs to differential drive
    M = r*np.array([[0.5, 0.5],
                    [1./L, -1/L]])
    u = np.linalg.inv(M) @ np.array([[v_d], [w_d]])
    print("Differential drive input:\n", u)

    # Check the resulting velocities
    ur = u.item(0)
    ul = u.item(1)
    v = r/2*(ur+ul)
    w = r/L*(ur-ul)
    print("Diff drive, resulting velocities: v = ", v, ", w = ", w)

    ### Bicycle model
    # Define problem parameters
    L = 0.2

    # Solve for the desired inputs
    phi = np.arctan(L*w_d/v_d)
    print("Bicycle, phi = ", phi)
    print("Bicycle check, w = ", v_d/L*np.tan(phi))

if __name__ == "__main__":
    #one_point_five_projection()
    two_point_four_velocity_control()
