"""Unitary completion and Algorithm 1 of Session 1.

Four of the five Sigma basis elements are not unitary, so no circuit builds
them directly. Unitary completion fixes that with one ancilla:

    U_l = [[A_l, A_l^c], [A_l^c, A_l]]  is unitary,
    U_l |0> |psi> = |0> A_l |psi> + |1> A_l^c |psi>.

A_l^c has no tensor product form, so U_l factors as U_{l,1} U_{l,2} with

    U_{l,1} = [[I - D, D], [D, I - D]],  D = A_l A_l^T,
    U_{l,2} = sigma_x (x) Abar_l.

D is diagonal with 0s and 1s, so U_{l,1} is one multi-controlled X gate.
"""
import numpy as np
from qiskit import QuantumCircuit, QuantumRegister, transpile
from qiskit.quantum_info import Operator

from .sigma import PRETTY, completion_matrix, sigma_matrix

# Which single-qubit gate the completion Abar_l needs at each letter.
COMPLETION_LETTER = {"I": "I", "+": "X", "-": "X", "P": "I", "M": "I"}

# Row bit of each letter. This is the control state of the multi-controlled X,
# because sigma_p sigma_p^T is |0><0| for + and P, and |1><1| for - and M.
ROW_BIT = {"+": 0, "-": 1, "P": 0, "M": 1}


def wire_of(p, n):
    """Wire that carries term position p. Position 0 is the most significant.

    Qiskit puts the most significant qubit on the highest wire, so position 0
    lands on wire n-1. This is the rule Qiskit uses for a Pauli label.
    """
    return n - 1 - p


def sigma_term_circuit(term, name=None):
    """Algorithm 1. Circuit for U_l on registers (q, a1).

    a1 is the unitary completion ancilla. It sits on the highest wire, so
    Qiskit treats it as the most significant qubit of U_l. That makes
    Operator(qc) the plain block matrix [[A_l, A_l^c], [A_l^c, A_l]].
    """
    n = len(term)
    q = QuantumRegister(n, "q")
    a1 = QuantumRegister(1, "a1")
    qc = QuantumCircuit(q, a1, name=name or "U[%s]" % term)

    # Steps 1 and 2: the unitary completion Abar_l, as single-qubit gates.
    # Step 3: the sigma_x factor of U_{l,2} acts on the ancilla.
    qc.x(a1[0])
    for p, letter in enumerate(term):
        if COMPLETION_LETTER[letter] == "X":
            qc.x(q[wire_of(p, n)])

    # Steps 4 and 5: U_{l,1} is one C^{n-k}X on the ancilla, where k counts
    # the identity letters. Every other letter contributes one control, open
    # for row bit 0 and closed for row bit 1.
    pairs = sorted((wire_of(p, n), ROW_BIT[letter])
                   for p, letter in enumerate(term) if letter != "I")
    if pairs:
        # ctrl_state bit j belongs to the j-th qubit of the control list.
        state = sum(bit << j for j, (_, bit) in enumerate(pairs))
        qc.mcx([q[w] for w, _ in pairs], a1[0], ctrl_state=state)
    else:
        qc.x(a1[0])          # A_l = I, so U_{l,1} = I and the two X gates cancel
    return qc


def pauli_term_circuit(pauli_string):
    """Circuit for one Pauli term, on the same register layout as a Sigma term."""
    n = len(pauli_string)
    q = QuantumRegister(n, "q")
    a1 = QuantumRegister(1, "a1")
    qc = QuantumCircuit(q, a1, name="P[%s]" % pauli_string)
    # A Pauli label and a Sigma term use the same rule: position p -> wire n-1-p.
    for p, letter in enumerate(pauli_string):
        w = wire_of(p, n)
        if letter == "X":
            qc.x(q[w])
        elif letter == "Y":
            qc.y(q[w])
        elif letter == "Z":
            qc.z(q[w])
    return qc


def circuit_matrix(qc):
    """Matrix of qc, in Qiskit's own qubit order. No bit reversal."""
    return Operator(qc).data


def cost(qc):
    """(CX count, depth) after transpiling to u and cx."""
    tq = transpile(qc, basis_gates=["u", "cx"], optimization_level=1)
    return tq.count_ops().get("cx", 0), tq.depth()


def check_term_circuit(term):
    """(max error, is unitary) against U_l = [[A_l, A_l^c], [A_l^c, A_l]]."""
    n = len(term)
    A = sigma_matrix(term)
    Ac = completion_matrix(term) - A
    want = np.block([[A, Ac], [Ac, A]])
    got = circuit_matrix(sigma_term_circuit(term)).real
    unitary = np.allclose(got @ got.conj().T, np.eye(2 ** (n + 1)))
    return np.abs(got - want).max(), unitary


def explain_term(term):
    """Lines that trace Algorithm 1 for one term, for printing."""
    n = len(term)
    out = ["Abar_l = " + " x ".join(COMPLETION_LETTER[w] for w in term).replace("X", "sx"),
           "k = number of identity factors = %d" % term.count("I"),
           "so U_{l,1} needs a C^%dX gate" % (n - term.count("I")),
           "",
           "letter -> wire:"]
    for p, letter in enumerate(term):
        out.append("   position %d  %-5s -> q%d" % (p, PRETTY[letter], wire_of(p, n)))
    return "\n".join(out)
