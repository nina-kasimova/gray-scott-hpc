# cython: boundscheck=False

import numpy as np

def update_cython(double[:, :] A,
                  double[:, :] B,
                  double DA, double DB, double f, double k, double delta_t):
    cdef int N = A.shape[0]
    cdef int M = A.shape[1]
    cdef int i, j, im1, ip1, jm1, jp1
    cdef double a, b, lap_a, lap_b, reaction

    cdef double[:, :] A_new = np.empty((N, M), dtype=np.float64)
    cdef double[:, :] B_new = np.empty((N, M), dtype=np.float64)

    for i in range(N):
        im1 = N - 1 if i == 0 else i - 1
        ip1 = 0 if i == N - 1 else i + 1

        for j in range(M):
            jm1 = M - 1 if j == 0 else j - 1
            jp1 = 0 if j == M - 1 else j + 1

            a = A[i, j]
            b = B[i, j]

            lap_a = A[im1, j] + A[ip1, j] + A[i, jm1] + A[i, jp1] - 4.0 * a
            lap_b = B[im1, j] + B[ip1, j] + B[i, jm1] + B[i, jp1] - 4.0 * b

            reaction = a * b * b

            A_new[i, j] = a + (DA * lap_a - reaction + f * (1.0 - a)) * delta_t
            B_new[i, j] = b + (DB * lap_b + reaction - (k + f) * b) * delta_t

    return np.asarray(A_new), np.asarray(B_new)