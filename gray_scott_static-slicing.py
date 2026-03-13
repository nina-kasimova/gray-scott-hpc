import numpy as np
# import matplotlib.pyplot as pl
import time
# ============ define relevant functions =============

# an efficient function to compute a mean over neighboring cells




def apply_laplacian(mat):
    neigh_mat = -4 * mat

    neigh_mat[1:, :]  += mat[:-1, :]   # up
    neigh_mat[:-1, :] += mat[1:, :]    # down
    neigh_mat[:, 1:]  += mat[:, :-1]   # left
    neigh_mat[:, :-1] += mat[:, 1:]    # right

    neigh_mat[0, :]  += mat[-1, :]
    neigh_mat[-1, :] += mat[0, :]
    neigh_mat[:, 0]  += mat[:, -1]
    neigh_mat[:, -1] += mat[:, 0]

    return neigh_mat

# Define the update formula for chemicals A and B


def update(A, B, DA, DB, f, k, delta_t):
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

# def draw(A, B):
#     """return the matplotlib artists for animation"""
#     fig, ax = pl.subplots(1,2,figsize=(5.65,3))
#     imA = ax[0].imshow(A, animated=True,vmin=0,cmap='Greys')
#     imB = ax[1].imshow(B, animated=True,vmax=1,cmap='Greys')
#     ax[0].axis('off')
#     ax[1].axis('off')
#     ax[0].set_title('A')
#     ax[1].set_title('B')
#
#     return fig, imA, imB

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

# intialize the chemical concentrations
A, B = get_initial_A_and_B(N)


#
N_sizes = [200,400,600, 800, 1000,1200, 1600]
N_steps = 1000
timings=[]
for N in N_sizes:
    A, B = get_initial_A_and_B(N)
    start = time.time()
    for step in range(N_steps):
        A, B = update(A, B, DA, DB, f, k, delta_t)
    end = time.time()
    timings.append(end - start)
    print(f"N={N}, grid={N}x{N}, steps={N_steps}, time={end - start:.3f}s")
print(timings)

# N_simulation_steps = 1000
#
# for step in range(N_simulation_steps):
#     A, B = update(A, B, DA, DB, f, k, delta_t)



# draw(A, B)
#
# # show the result
# pl.show()
