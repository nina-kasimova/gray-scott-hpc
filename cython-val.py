import numpy as np
import time
from gray_scott_cython import update_cython


def apply_laplacian(mat):
    """This function applies a discretized Laplacian
    in periodic boundary conditions to a matrix
    """
    neigh_mat = -4*mat.copy()
    neighbors = [
                    ( 1.0,  (-1, 0) ),
                    ( 1.0,  ( 0,-1) ),
                    ( 1.0,  ( 0, 1) ),
                    ( 1.0,  ( 1, 0) ),
                ]

    for weight, neigh in neighbors:
        neigh_mat += weight * np.roll(mat, neigh, (0,1))

    return neigh_mat

def update(A, B, DA, DB, f, k, delta_t):
    """Apply the Gray-Scott update formula"""
    # compute the diffusion part of the update
    diff_A = DA * apply_laplacian(A)
    diff_B = DB * apply_laplacian(B)

    # Apply chemical reaction
    reaction = A * B ** 2
    diff_A -= reaction
    diff_B += reaction

    # Apply birth/death
    diff_A += f * (1 - A)
    diff_B -= (k + f) * B

    A += diff_A * delta_t
    B += diff_B * delta_t

    return A, B

def get_initial_A_and_B(N, random_influence = 0.2):
    """get the initial chemical concentrations"""
    # get initial homogeneous concentrations
    A = (1-random_influence) * np.ones((N,N))
    B = np.zeros((N,N))

    # put some noise on there
    A += random_influence * np.random.random((N,N))
    B += random_influence * np.random.random((N,N))

    # get center and radius for initial disturbance
    N2, r = N//2, 50

    # apply initial disturbance
    A[N2-r:N2+r, N2-r:N2+r] = 0.50
    B[N2-r:N2+r, N2-r:N2+r] = 0.25

    return A, B

# =========== define model parameters ==========

# update in time
delta_t = 1.0

# Diffusion coefficients
DA = 0.16
DB = 0.08

# define birth/death rates
f = 0.060
k = 0.062

# grid size
N = 200

N_steps = 1000



np.random.seed(42)
A_start, B_start = get_initial_A_and_B(N)

# original
A_np, B_np = A_start.copy(), B_start.copy()
for step in range(N_steps):
    A_np, B_np = update(A_np, B_np, DA, DB, f, k, delta_t)

# cython
A_cy, B_cy = A_start.copy(), B_start.copy()
for step in range(N_steps):
    A_cy, B_cy = update_cython(A_cy, B_cy, DA, DB, f, k, delta_t)


print(f"A match: {np.allclose(A_np, A_cy, atol=1e-12)}")
print(f"B match: {np.allclose(B_np, B_cy, atol=1e-12)}")
print(f"A max diff: {np.max(np.abs(A_np - A_cy)):.2e}")
print(f"B max diff: {np.max(np.abs(B_np - B_cy)):.2e}")
