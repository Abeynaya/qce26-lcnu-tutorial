"""Automatic LCNU extraction by recursive halving.

This is the same move the hand derivations in decompose.py make, run as an
algorithm instead of by hand. Split a matrix on its most significant qubit:

    B = [[B00, B01], [B10, B11]]
      = P (x) B00 + M (x) B11 + (+) (x) B01 + (-) (x) B10

where P = sigma_+ sigma_-, M = sigma_- sigma_+. A zero block contributes
nothing, and each level halves the matrix, so the depth is log2 of the size.

The diagonal pair is where the terms are won or lost. Because P + M = I, for
ANY common part C:

    P (x) B00 + M (x) B11 = I (x) C + P (x) (B00 - C) + M (x) (B11 - C).

Picking C well is the whole game, and it is what lets the recursion cancel:
A' = -T + two corner entries is exactly the C = B00 choice, whose residual
B11 - B00 holds only those corners. Three candidates are tried at each level
and the one giving the fewest terms wins. A memo keyed on block content
collapses the repeated subproblems, so the search is cheap on a structured
matrix.

Term counts against the hand derivations:

    Poisson, any size    exact
    Neumann A', nx=64    16 against 15
    heat, nx=nt=32       33 against 31
    wave, nx=nt=32       36 against 38

Two limits worth knowing. The cost model is term count only: it ignores
lambda = sum |alpha_l| and the per-term circuit depth, so a term with fewer
identity letters (a wider C^mX) counts the same as a cheap one. And the memo
keys on dense block content, which is fine for a structured matrix but not
bounded for an arbitrary one.
"""
import numpy as np

from .sigma import dense

TOL = 1e-12

# Candidate common parts for the diagonal pair. Order is cosmetic; every one
# is tried. Each reduces the dimension by one, so the recursion terminates.
CANDIDATES = ("b00", "b11", "minmag")


def common_part(B00, B11, how, tol=TOL):
    """The common part C used in I (x) C + P (x) (B00-C) + M (x) (B11-C).

    b00     take B00 whole, so the P branch vanishes
    b11     take B11 whole, so the M branch vanishes
    minmag  take the smaller magnitude entrywise

    b00 and b11 choose one block for the whole level. minmag makes the same
    choice per entry, so at every entry one residual is exactly zero and no
    entry position goes down both residual branches. Magnitude is the rule
    because a zero has the smallest magnitude: where one block holds an entry
    and the other holds a zero, minmag keeps the zero and the entry stays a
    single nonzero instead of a cancelling pair.

    Two further candidates were measured and dropped. C = 0, the plain
    four-way split, never won on 29 test matrices, because minmag already
    gives C = 0 wherever either block holds a zero. Keeping only the entries
    where the blocks match won once, by one term, on a dense random matrix;
    it loses to minmag because a disagreeing entry goes down both residual
    branches.
    """
    if how == "b00":
        return B00.copy()
    if how == "b11":
        return B11.copy()
    if how == "minmag":
        return np.where(np.abs(B00) <= np.abs(B11), B00, B11)
    raise ValueError("unknown common part: %r" % (how,))


def merge_terms(terms, tol=1e-14):
    """Add the coefficients of repeated terms and drop the ones that cancel."""
    acc = {}
    for coeff, term in terms:
        acc[term] = acc.get(term, 0.0) + coeff
    return [(c, t) for t, c in acc.items() if abs(c) > tol]


def extract(A, candidates=CANDIDATES, tol=TOL):
    """LCNU decomposition of A as a list of (coefficient, term).

    Exact by construction: every rule used is an identity, so the result
    reconstructs A to machine precision. Not provably minimal, because the
    per-level choice is greedy over the candidate set.
    """
    A = np.asarray(A, dtype=float)
    n = int(round(np.log2(A.shape[0])))
    if 2 ** n != A.shape[0] or A.shape[0] != A.shape[1]:
        raise ValueError("A must be square of size 2^n")
    memo = {}

    def nz(B):
        return int((np.abs(B) > tol).sum())

    def go(B, m):
        if nz(B) == 0:
            return []
        if m == 0:
            return [(float(B[0, 0]), "")]
        key = (B.tobytes(), m)
        if key in memo:
            return memo[key]

        h = 1 << (m - 1)
        B00, B01, B10, B11 = B[:h, :h], B[:h, h:], B[h:, :h], B[h:, h:]

        if np.array_equal(B00, B11) and nz(B00):
            diag = [(c, "I" + t) for c, t in go(B00, m - 1)]      # C = B00 = B11
        else:
            diag = None
            for how in candidates:
                C = common_part(B00, B11, how, tol)
                trial = ([(c, "I" + t) for c, t in go(C, m - 1)]
                         + [(c, "P" + t) for c, t in go(B00 - C, m - 1)]
                         + [(c, "M" + t) for c, t in go(B11 - C, m - 1)])
                if diag is None or len(trial) < len(diag):
                    diag = trial

        out = (diag
               + [(c, "+" + t) for c, t in go(B01, m - 1)]
               + [(c, "-" + t) for c, t in go(B10, m - 1)])
        memo[key] = out
        return out

    return merge_terms(go(A, n))


def verify(A, terms, tol=1e-9):
    """Max absolute reconstruction error."""
    return float(np.abs(dense(terms) - np.asarray(A)).max())
