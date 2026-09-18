"""Discretized PDE matrices: Poisson, heat and wave.

Every space-time matrix stacks all time levels into one linear system, so a
quantum linear solver sees a single A u = b. The time index takes the high
bits of the row index and the space index takes the low bits.
"""
import numpy as np


def reference_poisson(nx):
    """The 1D Poisson matrix T: 2 on the diagonal, -1 off it."""
    T = np.zeros((nx, nx))
    for i in range(nx):
        T[i, i] = 2.0
        if i > 0:
            T[i, i - 1] = -1.0
        if i < nx - 1:
            T[i, i + 1] = -1.0
    return T


def laplacian_neumann(nx):
    """A' : 1D Laplacian with a zero-flux stencil at both ends.

    Equal to -T except at the two corners, which hold -1 instead of -2.
    """
    A = np.zeros((nx, nx))
    for i in range(nx):
        A[i, i] = -2.0
        if i > 0:
            A[i, i - 1] = 1.0
        if i < nx - 1:
            A[i, i + 1] = 1.0
    A[0, 0] = -1.0
    A[nx - 1, nx - 1] = -1.0
    return A


def heat_system(nx, nt, alpha=1.0, k=1.0, q=1.0, length=1.0, dt=0.01, u0=None):
    """Space-time matrix A and right-hand side b for the 1D heat equation.

    Flux q enters at x=0 and the end at x=l is insulated. Backward Euler in
    time, central differences in space. Block row 0 sets the initial
    condition, so the first diagonal block of the diffusion part is zero.
    """
    dx = length / (nx - 1)
    c = alpha * dt / dx ** 2
    Ap = laplacian_neumann(nx)
    if u0 is None:
        u0 = np.zeros(nx)

    N = nx * nt
    A = np.zeros((N, N))
    b = np.zeros(N)
    for j in range(nt):
        sl = slice(j * nx, (j + 1) * nx)
        if j == 0:
            A[sl, sl] = np.eye(nx)
            b[sl] = u0
        else:
            A[sl, sl] = np.eye(nx) - c * Ap
            A[sl, slice((j - 1) * nx, j * nx)] = -np.eye(nx)
            b[sl] = np.zeros(nx)
            b[j * nx] = q * dt / (k * dx)
    meta = dict(nx=nx, nt=nt, dx=dx, dt=dt, c=c, alpha=alpha, k=k, q=q, length=length)
    return A, b, meta


def wave_system(nx, nt, c=1.0, length=1.0, dt=0.05):
    """1D wave equation w_tt = c^2 w_xx, zero flux at both ends.

    Introduce v = dw/dt, then apply backward Euler to the first-order system:
        w_j - w_{j-1} = dt * v_j
        v_j - v_{j-1} = (c^2 dt / dx^2) A' w_j
    One time level holds u_j = [w_j ; v_j], of length 2 nx.

    This is not the discretization of the LCT paper, so the constant in its
    term count differs. The growth is the same.
    """
    dx = length / (nx - 1)
    k = c ** 2 * dt / dx ** 2
    m = 2 * nx
    B = np.block([[np.eye(nx), -dt * np.eye(nx)],
                  [-k * laplacian_neumann(nx), np.eye(nx)]])
    A = np.zeros((m * nt, m * nt))
    for j in range(nt):
        sl = slice(j * m, (j + 1) * m)
        A[sl, sl] = np.eye(m) if j == 0 else B
        if j:
            A[sl, slice((j - 1) * m, j * m)] = -np.eye(m)
    return A, dict(nx=nx, nt=nt, dt=dt, k=k)


# --- references used only to check the recursions ---------------------------
def reference_A1(nx, nt):
    """Block lower bidiagonal: I on the diagonal, -I below it."""
    N = nx * nt
    A = np.eye(N)
    for j in range(1, nt):
        A[j * nx:(j + 1) * nx, (j - 1) * nx:j * nx] = -np.eye(nx)
    return A


def reference_A2(nx, nt):
    """Block diagonal: zero in block 0, A' in every other block."""
    N = nx * nt
    A = np.zeros((N, N))
    Ap = laplacian_neumann(nx)
    for j in range(1, nt):
        A[j * nx:(j + 1) * nx, j * nx:(j + 1) * nx] = Ap
    return A


def nnz(A, tol=1e-12):
    return int((np.abs(A) > tol).sum())
