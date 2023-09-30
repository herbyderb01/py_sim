import time

import matplotlib.pyplot as plt
import numpy as np
from rtree import index
from scipy.spatial import KDTree


def nearest_ind_naive(points, list_ex, point) -> int:
    dist = np.inf
    ind = 0
    for index in list_ex:
        dist_i = np.linalg.norm(points[:,index:index+1]-point)
        if dist_i < dist:
            ind = index
            dist = dist_i

    return ind


def test_r_tree():

    # Create random points to insert
    points = np.random.random((2,10000)) * 1000.

    # Create the rtree
    tic = time.perf_counter()
    idx = index.Index()
    for k in range(points.shape[1]):
        idx.insert(k, (points[0,k], points[1,k], points[0,k], points[1,k]))
    toc = time.perf_counter()
    print(f"rtree construction in {toc-tic:0.4f} seconds")

    # Create a list holding everything
    list_ex = []
    tic = time.perf_counter()
    for k in range(points.shape[1]):
        list_ex.append(k)
    toc = time.perf_counter()
    print(f"list construction in {toc-tic:0.4f} seconds")


    # perform a single query
    desired = np.random.random((2,1))*1000.
    near = list(idx.nearest((desired.item(0),desired.item(1),desired.item(0),desired.item(1)), num_results=1))
    print("nearest = ", near)

    # Plot the results
    fig, ax = plt.subplots()
    ax.plot(points[0,:], points[1,:], 'bo')
    pnt = points[:,near[0]]
    ax.plot([desired.item(0)], [desired.item(1)], 'ro')
    ax.plot([pnt.item(0)], [pnt.item(1)], 'go')


    # perform a bunch of queries rtree
    query_pnts = np.random.random((2,10000))*1000.
    tic = time.perf_counter()
    for k in range(query_pnts.shape[1]):
        ind_close = idx.nearest((query_pnts[0,k], query_pnts[1,k], query_pnts[0,k], query_pnts[1,k]))
    toc = time.perf_counter()
    print(f"rtree query in {toc-tic:0.4f} seconds")

    # # perform a bunch of queries naive
    # tic = time.perf_counter()
    # for k in range(query_pnts.shape[1]):
    #     ind_close = nearest_ind_naive(points=points, list_ex=list_ex, point=query_pnts[:,k:k+1])
    # toc = time.perf_counter()
    # print(f"list query in {toc-tic:0.4f} seconds")

    # perform a bunch of queries using kdtree
    points_trans = points.transpose()
    tic = time.perf_counter()
    kdtree_ = KDTree(points_trans)
    for k in range(query_pnts.shape[1]):
        res = kdtree_.query(query_pnts[:,k].transpose())
    toc = time.perf_counter()
    print(f"list query - kd tree simple {toc-tic:0.4f} seconds")


    # # perform a bunch of queries using kdtree (very slow!!!!)
    # points_trans = points.transpose()
    # tic = time.perf_counter()
    # for k in range(query_pnts.shape[1]):
    #     kdtree_ = KDTree(points_trans)
    #     res = kdtree_.query(query_pnts[:,k].transpose())
    # toc = time.perf_counter()
    # print(f"list query - kd tree reconstruct {toc-tic:0.4f} seconds")


if __name__ == "__main__":
    test_r_tree()
    plt.show()