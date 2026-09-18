"""The Sigma basis, the term encoding, and Theorem 1.

A term is a string of n letters. Letter p names the tensor factor at term
position p, and position 0 is the most significant qubit.

    I  identity            accepts (row bit, col bit) in {(0,0), (1,1)}
    P  sigma_+ sigma_-      accepts (0, 0)
    +  sigma_+              accepts (0, 1)
    -  sigma_-              accepts (1, 0)
    M  sigma_- sigma_+      accepts (1, 1)
"""
from functools import reduce

import numpy as np

I2 = np.eye(2)
SP = np.array([[0.0, 1.0], [0.0, 0.0]])      # sigma_+ = |0><1|
SM = np.array([[0.0, 0.0], [1.0, 0.0]])      # sigma_- = |1><0|
PROJ0 = SP @ SM                              # sigma_+ sigma_- = |0><0|
PROJ1 = SM @ SP                              # sigma_- sigma_+ = |1><1|
SX = np.array([[0.0, 1.0], [1.0, 0.0]])      # sigma_x

SIGMA = {"I": I2, "+": SP, "-": SM, "P": PROJ0, "M": PROJ1}

# Unitary completion of each basis element (Session 1, completion table).
COMPLETION = {"I": I2, "+": SX, "-": SX, "P": I2, "M": I2}

PRETTY = {"I": "I", "+": "s+", "-": "s-", "P": "s+s-", "M": "s-s+"}

# (row bit, column bit) -> letter.  This table *is* Theorem 1.
ENTRY_LETTER = {(0, 0): "P", (0, 1): "+", (1, 0): "-", (1, 1): "M"}


def sigma_matrix(term):
    """Dense matrix of the tensor product named by `term`."""
    return reduce(np.kron, [SIGMA[w] for w in term])


def completion_matrix(term):
    """Dense matrix of Abar_l, the unitary completion of A_l."""
    return reduce(np.kron, [COMPLETION[w] for w in term])


def pretty(term):
    """Readable form of a term, for printing."""
    return " x ".join(PRETTY[w] for w in term)


def dense(terms):
    """Sum a list of (coefficient, term) into a dense matrix."""
    n = len(terms[0][1])
    out = np.zeros((2 ** n, 2 ** n), dtype=complex)
    for coeff, term in terms:
        out += coeff * sigma_matrix(term)
    return out.real if np.allclose(out.imag, 0) else out


def single_entry_term(r, c, n):
    """Theorem 1: the Sigma term of the matrix with one non-zero entry at (r, c)."""
    return "".join(ENTRY_LETTER[((r >> (n - 1 - p)) & 1, (c >> (n - 1 - p)) & 1)]
                   for p in range(n))
