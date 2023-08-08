import numpy as np
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt

if __name__ == "__main__":
    points = np.array([[0, 0], [0.5, 1], [0, 2], [1, 0], [1, 1], [1, 2],
                    [2, 0], [2, 1], [2, 2], [-1, -1], [-1, 3], [3,3], [3, -1]])

    vor = Voronoi(points)

    fig = voronoi_plot_2d(vor)
    plt.show(block=False)
    plt.show()