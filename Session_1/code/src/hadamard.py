"""State preparation and the two Hadamard tests.

A variational algorithm needs two kinds of number:

    <psi1| A_l |psi2>            and     <psi1| A_i^dag M A_j |psi2>,

with |psi1> = U|0>, |psi2> = V|0> and M unitary.

The plain Hadamard test needs a unitary in the middle, and A_l is not
unitary. Use U_l from circuits.py instead, then discard the branch where the
completion ancilla a1 reads 1.

Register order is (q, a1, a0), so wire significance rises from the system to
the ancillas. a1 is therefore the most significant qubit of U_l, which keeps
the block form exact in Qiskit's order.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.circuit.library import StatePreparation
from qiskit.quantum_info import Statevector

from .circuits import sigma_term_circuit


def state_of(prep):
    """State made by `prep`, in Qiskit's own qubit order."""
    return np.asarray(Statevector(prep).data)


def prep_from_vector(vec, name="prep"):
    """Circuit P with state_of(P) equal to vec / ||vec||."""
    v = np.asarray(vec, dtype=complex)
    v = v / np.linalg.norm(v)
    n = int(round(np.log2(len(v))))
    qc = QuantumCircuit(n, name=name)
    qc.append(StatePreparation(v), range(n))
    return transpile(qc, basis_gates=["u", "cx"], optimization_level=1)


def random_state_prep(n, seed, name="prep"):
    """A preparation circuit for a random complex state."""
    rng = np.random.default_rng(seed)
    v = rng.normal(size=2 ** n) + 1j * rng.normal(size=2 ** n)
    return prep_from_vector(v, name)


def test_registers(n):
    """(q, a1, a0). Wire order runs least to most significant."""
    q = QuantumRegister(n, "q")
    a1 = QuantumRegister(1, "a1")
    a0 = QuantumRegister(1, "a0")
    return q, a1, a0


# Controlled gates get reused thousands of times, so cache the synthesis.
_CTRL_CACHE = {}


def controlled(prep, label, ctrl_state):
    key = (id(prep), label, ctrl_state)
    if key not in _CTRL_CACHE:
        _CTRL_CACHE[key] = prep.to_gate(label=label).control(1, ctrl_state=ctrl_state)
    return _CTRL_CACHE[key]


def hadamard_test_single(term, prep1, prep2, imag=False):
    """Circuit whose P00 - P10 gives Re (or Im) of <psi1| A_l |psi2>."""
    n = len(term)
    q, a1, a0 = test_registers(n)
    qc = QuantumCircuit(q, a1, a0)

    qc.h(a0[0])                                                    # step 1
    if imag:
        qc.sdg(a0[0])
    qc.append(controlled(prep1, "U", 0), [a0[0], *q])              # step 2
    qc.append(controlled(prep2, "V", 1), [a0[0], *q])              # step 3
    qc.append(sigma_term_circuit(term).to_gate().control(1, ctrl_state=1),
              [a0[0], *q, a1[0]])                                  # step 4
    qc.h(a0[0])                                                    # step 5
    return qc


def hadamard_test_pair(term_i, term_j, prep1, prep2, prep_M=None, imag=False):
    """Circuit whose P00 - P10 gives Re (or Im) of <psi1| A_i^dag M A_j |psi2>.

    U_i goes on the a0 = 0 branch and U_j on the a0 = 1 branch. Both share
    the ancilla a1. M is controlled on a0 = 1, because an uncontrolled M
    would cancel through M^dag M = I.
    """
    n = len(term_i)
    q, a1, a0 = test_registers(n)
    qc = QuantumCircuit(q, a1, a0)

    qc.h(a0[0])
    if imag:
        qc.sdg(a0[0])
    qc.append(controlled(prep1, "U", 0), [a0[0], *q])
    qc.append(controlled(prep2, "V", 1), [a0[0], *q])
    qc.append(sigma_term_circuit(term_j).to_gate().control(1, ctrl_state=1),
              [a0[0], *q, a1[0]])
    if prep_M is not None:
        qc.append(controlled(prep_M, "M", 1), [a0[0], *q])
    qc.append(sigma_term_circuit(term_i).to_gate().control(1, ctrl_state=0),
              [a0[0], *q, a1[0]])
    qc.h(a0[0])
    return qc


def run_exact(qc):
    """P00 - P10 from the exact statevector. a0 is the top wire, a1 the next."""
    n = qc.num_qubits - 2
    probs = Statevector(qc).probabilities([n + 1, n])   # index = a0 + 2*a1
    return probs[0] - probs[1]


def inner_single(term, prep1, prep2):
    """<psi1| A_l |psi2> from two Hadamard tests."""
    return (run_exact(hadamard_test_single(term, prep1, prep2, imag=False))
            + 1j * run_exact(hadamard_test_single(term, prep1, prep2, imag=True)))


def inner_pair(term_i, term_j, prep1, prep2, prep_M=None):
    """<psi1| A_i^dag M A_j |psi2> from two Hadamard tests."""
    return (run_exact(hadamard_test_pair(term_i, term_j, prep1, prep2, prep_M, False))
            + 1j * run_exact(hadamard_test_pair(term_i, term_j, prep1, prep2, prep_M, True)))
