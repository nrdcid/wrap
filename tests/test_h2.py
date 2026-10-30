"""Every Clifford -> H2 decomposition must equal the source unitary up to a global phase."""

import numpy as np
import pytest

from wrap.backends.statevector import unitary
from wrap.ir import CLIFFORD, Gate, Qubit
from wrap.passes import CliffordToH2

a, b = Qubit("a"), Qubit("b")


def _kron_apply(gates, n):
    """Full 2^n unitary of a gate list on qubits [a, b][:n]."""
    order = [a, b][:n]
    U = np.eye(2**n, dtype=complex)
    for g in gates:
        u = unitary(g)
        if len(g.qubits) == n == 2 and g.qubits == (b, a):
            swap = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
            u = swap @ u @ swap
        elif len(g.qubits) == 1 and n == 2:
            u = np.kron(u, np.eye(2)) if g.qubits[0] == order[0] else np.kron(np.eye(2), u)
        U = u @ U
    return U


def _equal_up_to_phase(U, V):
    i = np.unravel_index(np.argmax(np.abs(U)), U.shape)
    phase = V[i] / U[i]
    return np.allclose(U * phase, V, atol=1e-9)


@pytest.mark.parametrize("name", sorted(CLIFFORD.gates))
def test_decomposition(name):
    n = CLIFFORD.gates[name]
    g = Gate(name, (a, b)[:n])
    lowered = CliffordToH2().compile_gate(g)
    assert all(l.name in {"RZ", "U1q", "ZZ"} for l in lowered)
    assert _equal_up_to_phase(_kron_apply(lowered, n), _kron_apply([g], n))
