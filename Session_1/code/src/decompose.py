"""Hand-derived LCNU recursions, plus the Pauli decomposition for comparison.

Every recursion uses one move. Split a matrix on its most significant qubit:

    B = [[B00, B01], [B10, B11]]
      = (s+s-) (x) B00 + (s+) (x) B01 + (s-) (x) B10 + (s-s+) (x) B11

Then apply two rules at each level:
  1. If B00 == B11, the two diagonal terms merge into I (x) B00.
  2. If an off-diagonal block is zero, its term disappears.

Each level halves the matrix, so the depth is log2 of the size. That is where
the polylogarithmic term count comes from.
"""
import numpy as np
from qiskit.quantum_info import SparsePauliOp

from .sigma import single_entry_term


def merge_terms(terms, tol=1e-14):
    """Add the coefficients of repeated terms and drop the ones that cancel."""
    acc = {}
    for coeff, term in terms:
        acc[term] = acc.get(term, 0.0) + coeff
    return [(c, w) for w, c in acc.items() if abs(c) > tol]


def lcnu_A1(t, s):
    """Block lower bidiagonal time-stepping matrix, as L (x) I_{2^s}.

    A1 = I^(t+s) - sum_{j=1..t} I^(j-1) (x) s- (x) s+^(t-j) (x) I^s.
    Gives t + 1 = log2(nt) + 1 terms, independent of nx.
    """
    terms = [(1.0, "I" * (t + s))]
    for j in range(1, t + 1):
        terms.append((-1.0, "I" * (j - 1) + "-" + "+" * (t - j) + "I" * s))
    return terms


def lcnu_poisson(s):
    """Session 1 recursion for the Poisson matrix T. 2s + 1 terms."""
    terms = [(2.0, "I" * s)]
    for j in range(1, s + 1):
        terms.append((-1.0, "I" * (j - 1) + "-" + "+" * (s - j)))
        terms.append((-1.0, "I" * (j - 1) + "+" + "-" * (s - j)))
    return terms


def lcnu_laplacian_neumann(s):
    """A' = -T + corner corrections. 2s + 3 terms.

    Theorem 1 supplies both corners. Entry (0,0) has all bits zero, so its
    term is all P. Entry (nx-1, nx-1) has all bits one, so its term is all M.
    """
    terms = [(-a, w) for a, w in lcnu_poisson(s)]
    terms.append((1.0, "P" * s))
    terms.append((1.0, "M" * s))
    return terms


def lcnu_A2(t, s):
    """(I_nt - |0><0|) (x) A'. 2 (2s + 3) terms.

    The second group only cancels time level 0. Those terms carry no matrix
    entries at all, which is the step a cover algorithm cannot reproduce.
    """
    out = []
    for coeff, term in lcnu_laplacian_neumann(s):
        out.append((coeff, "I" * t + term))
        out.append((-coeff, "P" * t + term))
    return out


def lcnu_heat(nx, nt, c, merge=True):
    """Hand decomposition of the space-time heat matrix.

    log2(nt) + 4 log2(nx) + 7 terms as derived, one fewer after merging.
    """
    s, t = int(round(np.log2(nx))), int(round(np.log2(nt)))
    assert 2 ** s == nx and 2 ** t == nt, "nx and nt must be powers of two"
    terms = lcnu_A1(t, s) + [(-c * a, w) for a, w in lcnu_A2(t, s)]
    return merge_terms(terms) if merge else terms


def lcnu_wave(nx, nt, dt, k, merge=True):
    """Hand decomposition of the space-time wave matrix of pdes.wave_system.

    (log2 nt + 1) + 2 + 2 (2 log2 nx + 3) = log2 nt + 4 log2 nx + 9 terms.
    """
    s, t = int(round(np.log2(nx))), int(round(np.log2(nt)))
    terms = lcnu_A1(t, s + 1)                           # L (x) I_{2nx}
    for cs, ws in [(1.0, "I" * t), (-1.0, "P" * t)]:    # I_nt - |0><0|
        terms.append((-dt * cs, ws + "+" + "I" * s))
        terms += [(-k * cs * ca, ws + "-" + wa)
                  for ca, wa in lcnu_laplacian_neumann(s)]
    return merge_terms(terms) if merge else terms


def lcnu_by_theorem1(A, tol=1e-12):
    """Baseline: one term per non-zero entry. Never uses the identity letter."""
    n = int(round(np.log2(A.shape[0])))
    rows, cols = np.nonzero(np.abs(A) > tol)
    return [(A[r, c], single_entry_term(int(r), int(c), n)) for r, c in zip(rows, cols)]


def pauli_lcu(A, tol=1e-12):
    """Pauli basis LCU decomposition, as a list of (coefficient, label)."""
    op = SparsePauliOp.from_operator(A).simplify(atol=tol)
    return list(zip(op.coeffs, [str(p) for p in op.paulis]))
