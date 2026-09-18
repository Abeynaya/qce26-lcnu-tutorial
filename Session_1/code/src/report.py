"""Tables and plots for the notebook. No new method lives here.

Set SAVE_FIGS to a directory and every plot also lands there as a PDF, for
the Beamer deck to include.
"""
import os

import numpy as np
import matplotlib.pyplot as plt

from . import autoextract
from .circuits import check_term_circuit, cost, pauli_term_circuit, sigma_term_circuit
from .decompose import (lcnu_by_theorem1, lcnu_heat, lcnu_poisson,
                        pauli_lcu)
from .pdes import heat_system, nnz, reference_poisson, wave_system
from .sigma import dense, pretty

SAVE_FIGS = os.environ.get("LCNU_FIGDIR")

# --- slide geometry --------------------------------------------------------
# The deck includes these PDFs at \linewidth of a 16:9 beamer frame, which
# pdflatex reports as 398.34pt. Build every figure at exactly that width and
# include it at exactly \linewidth, so the scale factor is 1 and a matplotlib
# point is a point on the slide. FONT_PT is then what the audience reads.
#
# Do not pass bbox_inches="tight" to savefig: it crops the saved PDF to the
# ink, which changes the width and breaks that correspondence. tight_layout
# already removes the slack.
SLIDE_W_IN = 398.3386 / 72.27       # \linewidth of the frame, in inches
# The figures used to land at about 4pt on the slide, because they were built
# wide and scaled down. 7pt is only a little larger than that but reads from
# the back of a room, and it leaves the panels their height.
FONT_PT = 7.0                       # ticks, axis labels and legends
TITLE_PT = 8.0                      # panel titles


def col_in(frac):
    """Width in inches of a beamer column of frac x \\textwidth.

    A figure inside a column is included at that column's \\linewidth, not the
    frame's, so it has to be built this narrow or the fonts scale down again.
    """
    return SLIDE_W_IN * frac

plt.rcParams.update({
    "font.size": FONT_PT,
    "axes.labelsize": FONT_PT,
    "axes.titlesize": TITLE_PT,
    "xtick.labelsize": FONT_PT,
    "ytick.labelsize": FONT_PT,
    "legend.fontsize": FONT_PT,
    "figure.titlesize": TITLE_PT,
    "lines.linewidth": 1.5,
    "lines.markersize": 4,
})


# matplotlib's own defaults, for a figure that keeps its original look and so
# opts out of the slide settings above. Use with plt.rc_context.
ORIGINAL_STYLE = {
    "font.size": 10.0,
    "font.weight": "normal",
    "axes.labelsize": 10.0,
    "axes.labelweight": "normal",
    "axes.titlesize": 12.0,
    "axes.titleweight": "normal",
    "xtick.labelsize": 10.0,
    "ytick.labelsize": 10.0,
    "legend.fontsize": 10.0,
    "lines.linewidth": 1.5,
    "lines.markersize": 6.0,
}


def save_fig(name, tight=False):
    """Write the current figure to SAVE_FIGS as a PDF, if it is set.

    tight=True crops the PDF to the ink, which breaks the scale-1 rule above.
    Only a figure built at some other width wants it.
    """
    if SAVE_FIGS:
        os.makedirs(SAVE_FIGS, exist_ok=True)
        path = os.path.join(SAVE_FIGS, name + ".pdf")
        plt.savefig(path, bbox_inches="tight" if tight else None)


_save = save_fig


# --- section 3 -------------------------------------------------------------
def show_terms(terms, A, label=""):
    """Print the term list, the reconstruction error and lambda."""
    err = np.abs(dense(terms) - A).max()
    print("%s%d terms,  reconstruction error %.1e,  lambda = %.4f"
          % (label, len(terms), err, sum(abs(a) for a, _ in terms)))
    print("\n  l   alpha_l      A_l")
    for i, (a, w) in enumerate(terms):
        print("  %2d  %+9.4f   %s" % (i, a, pretty(w)))


def check_recursions(pairs=((1, 1), (2, 2), (3, 2), (2, 3), (3, 3))):
    """Verify A1, A' and A2 against dense references over a grid of sizes."""
    from .decompose import lcnu_A1, lcnu_A2, lcnu_laplacian_neumann
    from .pdes import laplacian_neumann, reference_A1, reference_A2
    print("  nx   nt    A1 terms  (log2 nt + 1)   A2 terms  (4 log2 nx + 6)   max error")
    for s, t in pairs:
        nx, nt = 2 ** s, 2 ** t
        e1 = np.abs(dense(lcnu_A1(t, s)) - reference_A1(nx, nt)).max()
        e2 = np.abs(dense(lcnu_A2(t, s)) - reference_A2(nx, nt)).max()
        ea = np.abs(dense(lcnu_laplacian_neumann(s)) - laplacian_neumann(nx)).max()
        print("  %3d  %3d       %3d  %12d       %3d  %14d   %.1e"
              % (nx, nt, len(lcnu_A1(t, s)), t + 1,
                 len(lcnu_A2(t, s)), 4 * s + 6, max(e1, e2, ea)))


# --- section 4 -------------------------------------------------------------
def pauli_vs_lcnu(smax=4, tmax=4):
    """Term counts for Pauli, LCNU and the Theorem 1 baseline."""
    rows = []
    for s in range(1, smax + 1):
        for t in range(1, tmax + 1):
            nx, nt = 2 ** s, 2 ** t
            A, b, meta = heat_system(nx, nt)
            rows.append(dict(nx=nx, nt=nt, N=nx * nt, nnz=nnz(A),
                             pauli=len(pauli_lcu(A)),
                             lcnu=len(lcnu_heat(nx, nt, meta["c"])),
                             theorem1=len(lcnu_by_theorem1(A))))
    print(" nx   nt      N   nnz(A)   Pauli   hand-compute   Thm1   Pauli/LCNU")
    for r in rows:
        print("%3d  %3d  %5d   %6d   %5d   %12d   %4d   %8.1fx"
              % (r["nx"], r["nt"], r["N"], r["nnz"], r["pauli"], r["lcnu"],
                 r["theorem1"], r["pauli"] / r["lcnu"]))
    return rows


def plot_pauli_vs_lcnu(rows):
    fig, ax = plt.subplots(1, 2, figsize=(SLIDE_W_IN, 1.83))
    Ns = sorted({r["N"] for r in rows})
    # hand-compute keeps the C2 green and nnz the grey it has in plot_growth.
    for label, key, style, colour in [("Pauli LCU", "pauli", "o-", "C0"),
                                      ("LCNU", "lcnu", "s-", "C2"),
                                      ("nnz(A)", "nnz", "^--", "0.55")]:
        ax[0].semilogy(Ns, [max(r[key] for r in rows if r["N"] == N) for N in Ns],
                       style, color=colour, label=label)
    ax[0].set_xlabel("N = nx*nt")
    ax[0].set_ylabel("number of terms")
    ax[0].set_title("Heat Eqn", pad=3)

    # Same three curves as the left panel, so the two read the same way.
    nxs = [2 ** s for s in range(1, 9)]
    pois = [reference_poisson(nx) for nx in nxs]
    ax[1].semilogy(nxs, [len(pauli_lcu(P)) for P in pois], "o-", color="C0",
                   label="Pauli LCU")
    ax[1].semilogy(nxs, [2 * int(np.log2(nx)) + 1 for nx in nxs], "s-", color="C2",
                   label="LCNU")
    ax[1].semilogy(nxs, [nnz(P) for P in pois], "^--", color="0.55", label="nnz(A)")
    ax[1].set_xlabel("N = nx")
    ax[1].set_title("Poisson Eqn", pad=3)
    for a in ax:
        a.grid(alpha=0.3)
    # One legend in its own row under the panels. Inside an axes it covers the
    # curves, and both panels carry the same three series anyway.
    h, lab = ax[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False,
               handlelength=1.8, columnspacing=2.2, borderaxespad=0.1)
    plt.tight_layout(rect=(0, 0.11, 1, 1))
    _save("pauli_vs_lcnu")
    plt.show()


# --- section 5 -------------------------------------------------------------
def check_all_terms(terms, n_random=200, seed=0):
    """Check U_l for every term, then for a batch of random terms."""
    worst = 0.0
    print("  l   A_l                            err      unitary")
    for i, (a, w) in enumerate(terms):
        err, uni = check_term_circuit(w)
        worst = max(worst, err)
        print("  %2d  %-28s  %.1e   %s" % (i, pretty(w), err, uni))
    print("\nworst error over all %d terms: %.2e" % (len(terms), worst))
    n = len(terms[0][1])
    rng = np.random.default_rng(seed)
    wr = max(check_term_circuit("".join(rng.choice(list("I+-PM"), size=n)))[0]
             for _ in range(n_random))
    print("worst error over %d random %d-qubit terms: %.2e" % (n_random, n, wr))


def resource_table(nx, nt):
    """Gate cost and circuit count for the Sigma and Pauli decompositions."""
    A, b, meta = heat_system(nx, nt)
    sig = [cost(sigma_term_circuit(w)) for _, w in lcnu_heat(nx, nt, meta["c"])]
    pau = [cost(pauli_term_circuit(p)) for _, p in pauli_lcu(A)]
    print("nx=%d, nt=%d, n=%d qubits" % (nx, nt, int(round(np.log2(nx * nt)))))
    print("  %-20s %6s %8s %9s %9s %12s"
          % ("basis", "terms", "max CX", "total CX", "max depth", "O(n_l^2) tests"))
    for label, c in [("Sigma basis LCNU", sig), ("Pauli basis LCU", pau)]:
        print("  %-20s %6d %8d %9d %9d %12d"
              % (label, len(c), max(x for x, _ in c), sum(x for x, _ in c),
                 max(d for _, d in c), len(c) ** 2))
    print("  circuits saved: %.1fx\n" % ((len(pau) / len(sig)) ** 2))


# --- section 8 -------------------------------------------------------------
def autoextract_sweep():
    """Recursive and hand-compute term counts for all three PDE families."""
    out = {"Poisson": [], "heat": [], "wave": []}
    for s in range(3, 9):
        nx = 2 ** s
        out["Poisson"].append(dict(label="nx=%d" % nx, A=reference_poisson(nx),
                                   hand=len(lcnu_poisson(s))))
    for s in range(2, 6):
        nx = nt = 2 ** s
        A, b, meta = heat_system(nx, nt)
        out["heat"].append(dict(label="nx=nt=%d" % nx, A=A,
                                hand=len(lcnu_heat(nx, nt, meta["c"]))))
    for s in range(2, 6):
        nx = nt = 2 ** s
        A, meta = wave_system(nx, nt)
        # LCT count for the wave equation: log nt + 1 + 2(2 log 2nx + 4)
        out["wave"].append(dict(label="nx=nt=%d" % nx, A=A,
                                hand=s + 1 + 2 * (2 * (s + 1) + 4)))
    for fam in out:
        for row in out[fam]:
            terms = autoextract.extract(row["A"])
            row.update(N=row["A"].shape[0], nnz=nnz(row["A"]), auto=len(terms),
                       err=autoextract.verify(row["A"], terms))

    print("%-8s %-10s %6s %7s   %9s %13s    %s"
          % ("family", "size", "N", "nnz", "recursive", "hand-compute", "error"))
    print("-" * 72)
    for fam in ("Poisson", "heat", "wave"):
        for r in out[fam]:
            print("%-8s %-10s %6d %7d   %9d %13d    %.0e"
                  % (fam, r["label"], r["N"], r["nnz"], r["auto"], r["hand"],
                     r["err"]))
        print("-" * 72)
    print("recursive    = src/autoextract.py, halving with a chosen common part")
    print("hand-compute = src/decompose.py, and the LCT formula for wave")
    return out


FAMILIES = ("Poisson", "heat", "wave")


def _qubits(rows):
    return np.array([int(round(np.log2(r["N"]))) for r in rows], dtype=float)


def _title(fam):
    """Same convention as plot_pauli_vs_lcnu. nt = nx goes in the shared x label."""
    return "%s Eqn" % fam.capitalize()


def plot_growth(data):
    """Linear y against the qubit count, so linear in log N is a straight line."""
    fig, axes = plt.subplots(1, 3, figsize=(SLIDE_W_IN, 2.18), sharey=True)
    # No fitted curve: the recursive count is concave over this range, so a
    # straight-line fit reads steeper than the trend it is meant to show.
    fits = [("nnz(A)", "nnz", "^--", "0.55"),
            ("recursive", "auto", "o-", "C0"),
            ("hand-compute", "hand", "s-", "C2")]
    for ax, fam in zip(axes, FAMILIES):
        rows = data[fam]
        n = _qubits(rows)
        for label, key, style, colour in fits:
            y = np.array([r[key] for r in rows], dtype=float)
            ax.plot(n, y, style, color=colour, label=label)
        ax.set_xticks(n)
        ax.set_title(_title(fam), pad=3)
        ax.set_yscale("log")          # nnz(A) dwarfs both counts on a linear axis
        ax.set_yticks([1e1, 1e2, 1e3, 1e4])
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel("terms")
    # One x label, on the middle panel. tight_layout does not account for
    # fig.supxlabel, which leaves a gap under the panels on the slide.
    axes[1].set_xlabel("qubits   n = log2 N")
    # One legend in its own row under the panels, for the same reason as in
    # plot_pauli_vs_lcnu: inside an axes it covers the curves.
    h, lab = axes[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="lower center", ncol=3, frameon=False,
               handlelength=1.8, columnspacing=2.2, borderaxespad=0.1)
    plt.tight_layout(rect=(0, 0.10, 1, 1))
    save_fig("autoextract_growth")
    plt.show()


def plot_autoextract(data):
    """Term count against the qubit count, then the gap to hand-compute."""
    plot_growth(data)

    # Report the gap and the per-step increment. Both are honest over 4 points;
    # a fitted slope is not, because the recursive curve is concave here.
    print("gap to hand-compute, and the increment per doubling of n_x\n")
    print("  family    size        recursive  hand   gap   d(rec)  d(hand)")
    for fam in FAMILIES:
        rows = data[fam]
        for k, r in enumerate(rows):
            da = r["auto"] - rows[k - 1]["auto"] if k else None
            dh = r["hand"] - rows[k - 1]["hand"] if k else None
            print("  %-8s  %-10s %8d %6d %5d   %5s   %5s"
                  % (fam, r["label"], r["auto"], r["hand"],
                     r["auto"] - r["hand"],
                     "-" if da is None else da, "-" if dh is None else dh))
    print("\nThe increments converge, so the two counts stay within a couple")
    print("of terms rather than diverging.")
