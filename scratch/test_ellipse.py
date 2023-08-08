"""test_ellipse.py tests the generation of an ellipse as well as the sampling of points within that ellipse
"""

from py_sim.tools.sim_types import TwoDimArray
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Ellipse
import py_sim.path_planning.rrt_planner as rrt

def in_ellipse(point: TwoDimArray, center: TwoDimArray, a: float, b: float, alpha: float) -> bool:
    p = TwoDimArray(vec=point.state-center.state) # translated point so that the center is the adjusted origin

    c_alpha = np.cos(alpha)
    s_alpha = np.sin(alpha)
    v = (p.x*c_alpha + p.y*s_alpha)**2 / a**2 + (p.x*s_alpha-p.y*c_alpha)**2/b**2 - 1

    return bool(v < 0)


def main() -> None:
    """ Tests out the ellipse by doing the following:
        1. Drawing the ellipse
        2. Sampling randomly within a rectangle containing the ellipse
        3. Plot samples by color based on whether or not they are in the ellipse
    """
    # Define the ellipse
    p1 = TwoDimArray(x=np.random.random()*5, y=np.random.random()*5)
    p2 = TwoDimArray(x=np.random.random()*5, y=np.random.random()*5)
    # p1 = TwoDimArray(x=-2., y=-2.)
    # p2 = TwoDimArray(x=8., y=8.)
    c_min = np.linalg.norm(p1.state-p2.state)
    c_best = c_min + 1. # Some random size bigger then c_min
    a = c_best/2 # Major axis
    b = np.sqrt(c_best**2 - c_min**2) / 2 # Minor axis
    center = TwoDimArray(vec=(p1.state + p2.state)/2.)
    alpha = np.arctan2(p2.y-p1.y, p2.x-p1.x)

    # a = 8
    # b = 2
    # center = TwoDimArray(x=0., y=0.)
    # alpha = 0.


    ###### Plotting an ellipse #######
    fig, ax = plt.subplots()




    ###### Sample many points #######
    # Create bounding box
    c_a = np.cos(alpha)
    s_a = np.sin(alpha)
    A = c_a**2/a**2 + s_a**2/b**2
    B = 2*c_a*s_a*(1/a**2 - 1/b**2)
    C = s_a**2/a**2+c_a**2/b**2
    F = -1.
    x_diff = np.sqrt((4.*C*F)/(B**2-4*A*C))
    y_diff =np.sqrt((4.*A*F)/(B**2-4*A*C))


    #X = rrt.StateSpace(x_lim=(center.x-a, center.x+a), y_lim=(center.y-a, center.y+a))
    X = rrt.StateSpace(x_lim=(center.x-x_diff, center.x+x_diff), y_lim=(center.y-y_diff, center.y+y_diff))


    # Create sampling vector
    x_in_vec: list[float] = []
    y_in_vec: list[float] = []
    x_out_vec: list[float] = []
    y_out_vec: list[float] = []

    N = 1000
    for _ in range(N):
        state = rrt.sample(X=X)
        #state = TwoDimArray(x = -4.1, y=0.)

        if in_ellipse(point=state, center=center, a=a, b=b, alpha=alpha):
            x_in_vec.append(state.x)
            y_in_vec.append(state.y)
        else:
            x_out_vec.append(state.x)
            y_out_vec.append(state.y)

    ##### Plot the samples by color ######
    ax.plot(x_in_vec, y_in_vec, 'go')
    ax.plot(x_out_vec, y_out_vec, 'ro')

    # Plot the points of interest
    ax.plot([p1.x], [p1.y], 'ko')
    ax.plot([p2.x], [p2.y], 'ko')

    # Plot the ellipse
    ellipse = patches.Ellipse(xy=(center.x, center.y), width=a*2, height=b*2, angle=np.rad2deg(alpha), visible=True, fill=False)
    ax.add_patch(ellipse)
    #ax.add_artist(ellipse)

    ###### Leave the plot displayed ######
    # Display the plot
    ax.set_aspect('equal', 'box')

    ax.set_xlim(-10, 10)
    ax.set_ylim(-10, 10)

    plt.show(block=False)
    plt.show()

def test_code():
    NUM = 250

    ells = [Ellipse(xy=np.random.rand(2) * 10,
                    width=np.random.rand(), height=np.random.rand(),
                    angle=np.random.rand() * 360)
            for i in range(NUM)]

    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})
    for e in ells:
        ax.add_artist(e)
        e.set_clip_box(ax.bbox)
        e.set_alpha(np.random.rand())
        e.set_facecolor(np.random.rand(3))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    plt.show()

def test_code_modified():
    NUM = 1

    e = Ellipse(xy=np.random.rand(2) * 10,
                    width=np.random.rand(), height=np.random.rand(),
                    angle=np.random.rand() * 360)


    fig, ax = plt.subplots(subplot_kw={'aspect': 'equal'})

    ax.add_artist(e)
    #e.set_clip_box(ax.bbox)
    #e.set_alpha(np.random.rand())
    #e.set_facecolor(np.random.rand(3))

    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)

    plt.show()




if __name__ == "__main__":
    main()
    #test_code()
    #test_code_modified()