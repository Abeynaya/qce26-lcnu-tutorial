"""Helpers for the LCNU hands-on notebook.

| module | contents |
|---|---|
| `sigma` | the Sigma basis, the term encoding, Theorem 1 |
| `pdes` | heat, wave and Poisson matrices |
| `decompose` | the hand recursions, plus the Pauli decomposition |
| `circuits` | unitary completion and Algorithm 1 |
| `hadamard` | state preparation and the two Hadamard tests |
| `vqls` | the global and local cost functions |
| `autoextract` | automatic extraction, the cover algorithm |
| `report` | the tables and the plots used in the notebook |

Qubit order follows Qiskit. Term position `p` goes on wire `n-1-p`, which is
the rule Qiskit uses for a Pauli label. No bit reversal happens anywhere.
"""
