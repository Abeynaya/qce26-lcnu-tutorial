"""The VQLS global and local cost functions, built from Hadamard tests.

With |b> = U|0>, |psi> = V(theta)|0> and |phi> = A|psi>:

    C_g = 1 - |<b|phi>|^2 / <phi|phi>,
    <phi|phi>   = sum_ij alpha_i conj(alpha_j) beta_ij,
    |<b|phi>|^2 = sum_ij alpha_i conj(alpha_j) gamma_ij,

    beta_ij  = <psi| A_j^dag A_i |psi>,
    gamma_ij = c_i conj(c_j),   c_i = <b| A_i |psi>.

gamma factorizes, so it costs n_l single-operator tests. beta does not, and
costs n_l (n_l + 1) / 2 two-operator tests. The local cost function adds a
factor of n through delta_ijk.
"""
from functools import reduce

import numpy as np
from qiskit import QuantumCircuit

from .circuits import circuit_matrix, wire_of
from .hadamard import inner_pair, inner_single, state_of


def ansatz(theta, n, layers):
    """Real hardware-efficient ansatz. Ry rotations and a CX ladder."""
    qc = QuantumCircuit(n, name="V")
    idx = 0
    for p in range(n):
        qc.ry(theta[idx], p)
        idx += 1
    for _ in range(layers):
        for p in range(n - 1):
            qc.cx(p, p + 1)
        for p in range(n):
            qc.ry(theta[idx], p)
            idx += 1
    return qc


def n_params(n, layers):
    return n * (layers + 1)


def gamma_from_tests(terms, prep_b, prep_psi):
    """gamma_ij = c_i conj(c_j) with c_i = <b| A_i |psi>. n_l single tests."""
    c = np.array([inner_single(w, prep_b, prep_psi) for _, w in terms])
    return np.outer(c, c.conj())


def beta_from_tests(terms, prep_psi):
    """beta_ij = <psi| A_j^dag A_i |psi>. Hermitian, so fill the lower half."""
    nl = len(terms)
    term_ops = [t for _, t in terms]
    beta = np.zeros((nl, nl), dtype=complex)
    for i in range(nl):
        for j in range(i, nl):
            # inner_pair(x, y, ...) gives <psi| A_x^dag A_y |psi>, so pass (j, i).
            beta[i, j] = inner_pair(term_ops[j], term_ops[i], prep_psi, prep_psi)
            beta[j, i] = beta[i, j].conjugate()
    return beta


def beta_exact(terms, prep_psi):
    """beta from dense linear algebra, to check the tests against."""
    from .sigma import sigma_matrix
    psi = state_of(prep_psi)
    return np.array([[psi.conj() @ (sigma_matrix(wj).T.conj() @ sigma_matrix(wi) @ psi)
                      for _, wj in terms] for _, wi in terms]).T


def cost_global_from_tests(terms, prep_b, prep_psi, verbose=False):
    nl = len(terms)
    alpha = np.array([a for a, _ in terms], dtype=complex)
    beta = beta_from_tests(terms, prep_psi)
    gamma = gamma_from_tests(terms, prep_b, prep_psi)
    phi_norm2 = np.real(alpha @ beta @ alpha.conj())
    overlap2 = np.real(alpha @ gamma @ alpha.conj())
    if verbose:
        print("  single-operator tests: %d" % (2 * nl))
        print("  two-operator tests:    %d" % (2 * nl * (nl + 1) // 2))
        print("  <phi|phi>   = %.8f" % phi_norm2)
        print("  |<b|phi>|^2 = %.8f" % overlap2)
    return 1.0 - overlap2 / phi_norm2


def cost_global_exact(A, b, prep_psi):
    phi = A @ state_of(prep_psi)
    bhat = b / np.linalg.norm(b)
    return 1.0 - abs(np.vdot(bhat, phi)) ** 2 / np.vdot(phi, phi).real


def prep_M_local(prep_b, k, n):
    """M = U (Z_k tensor I) U^dag, as a circuit.

    Qiskit applies gates left to right, so the matrix of a circuit built as
    [X, Y, Z] is Z Y X. Append U^dag first to get U Z_k U^dag.
    """
    qc = QuantumCircuit(n, name="M%d" % k)
    qc.compose(prep_b.inverse(), range(n), inplace=True)
    qc.z(k)
    qc.compose(prep_b, range(n), inplace=True)
    return qc


def cost_local_from_tests(terms, prep_b, prep_psi, beta=None):
    nl, n = len(terms), len(terms[0][1])
    alpha = np.array([a for a, _ in terms], dtype=complex)
    term_ops = [t for _, t in terms]
    if beta is None:
        beta = beta_from_tests(terms, prep_psi)
    phi_norm2 = np.real(alpha @ beta @ alpha.conj())

    total = 0.0
    for k in range(n):
        pm = prep_M_local(prep_b, k, n)
        delta = np.zeros((nl, nl), dtype=complex)
        for i in range(nl):
            for j in range(i, nl):
                delta[i, j] = inner_pair(term_ops[j], term_ops[i], prep_psi, prep_psi, pm)
                delta[j, i] = delta[i, j].conjugate()
        total += np.real(alpha @ (beta + delta) @ alpha.conj()) / 2.0
    return 1.0 - total / (n * phi_norm2)


def cost_local_exact(A, prep_b, prep_psi):
    n = int(round(np.log2(A.shape[0])))
    phi = A @ state_of(prep_psi)
    Ub = circuit_matrix(prep_b)
    proj = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for k in range(n):
        # wire k is term position n-1-k, so Z sits at that position
        Zk = reduce(np.kron, [np.diag([1.0, -1.0]) if p == wire_of(k, n) else np.eye(2)
                              for p in range(n)])
        proj += (np.eye(2 ** n) + Zk) / 2.0
    H = np.eye(2 ** n) - proj / n
    num = np.real(phi.conj() @ (Ub @ H @ Ub.conj().T) @ phi)
    return num / np.vdot(phi, phi).real
