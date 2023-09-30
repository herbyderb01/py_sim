
import numpy as np


def test_eigenvector():
    A_mat = np.array([[2., 0., 0.],
                      [2., 2., 2.],
                      [3., 0., -1.]])

    vals, vecs = np.linalg.eig(A_mat)

    vec1 = A_mat @ vecs[:,[0]]
    vec2 = A_mat @ vecs[:,[1]]
    vec3 = A_mat @ vecs[:,[2]]


    print(vals)
    print(vecs)

    print(vec1)
    print(vec2)
    print(vec3)

if __name__ == "__main__":
    test_eigenvector()
