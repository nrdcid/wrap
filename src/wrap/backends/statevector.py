"""Reference state-vector emulator.

Executes a Program against a callback table, running callbacks *mid-circuit*
(so syndrome-driven corrections happen between gates).  Qubits are allocated
and freed dynamically; the state is a rank-n tensor of shape (2,)*n.

Noise: an optional stochastic Pauli channel applied after every gate
(Monte-Carlo), enough to demonstrate that QEC passes lower logical error.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import cos, sin
from typing import Callable, Mapping

import numpy as np

from ..callbacks import Callbacks
from ..ir import Gate, Kernel, Program, Qubit
from ..mstack import MeasurementStack

# ---- gate unitaries ------------------------------------------------------------

_I = np.eye(2, dtype=complex)
_X = np.array([[0, 1], [1, 0]], dtype=complex)
_Y = np.array([[0, -1j], [1j, 0]], dtype=complex)
_Z = np.array([[1, 0], [0, -1]], dtype=complex)
_H = np.array([[1, 1], [1, -1]], dtype=complex) / np.sqrt(2)
_S = np.array([[1, 0], [0, 1j]], dtype=complex)
_CX = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 0, 1], [0, 0, 1, 0]], dtype=complex)
_CZ = np.diag([1, 1, 1, -1]).astype(complex)


def _rz(t: float) -> np.ndarray:
    return np.array([[np.exp(-1j * t / 2), 0], [0, np.exp(1j * t / 2)]])


def _u1q(t: float, p: float) -> np.ndarray:
    n = cos(p) * _X + sin(p) * _Y
    return cos(t / 2) * _I - 1j * sin(t / 2) * n


def _zz() -> np.ndarray:
    return np.diag(np.exp(-1j * np.pi / 4 * np.array([1, -1, -1, 1])))


UNITARIES: Mapping[str, Callable[..., np.ndarray]] = {
    # toy
    "flip": lambda: _X,
    "mix": lambda: _H,
    "entangle": lambda: _CX,
    # clifford
    "X": lambda: _X,
    "Y": lambda: _Y,
    "Z": lambda: _Z,
    "H": lambda: _H,
    "S": lambda: _S,
    "CX": lambda: _CX,
    "CZ": lambda: _CZ,
    # h2
    "RZ": _rz,
    "U1q": _u1q,
    "ZZ": _zz,
}


def unitary(gate: Gate) -> np.ndarray:
    return UNITARIES[gate.name](*gate.params)


# ---- noise ---------------------------------------------------------------------


@dataclass
class PauliNoise:
    """After each gate, apply X/Y/Z to each touched qubit w.p. p/3 each."""

    p: float

    def sample(self, rng: np.random.Generator) -> np.ndarray | None:
        r = rng.random()
        if r < self.p / 3:
            return _X
        if r < 2 * self.p / 3:
            return _Y
        if r < self.p:
            return _Z
        return None


# ---- emulator ----------------------------------------------------------------


class DynamicStackError(RuntimeError):
    pass


@dataclass
class Emulator:
    noise: PauliNoise | None = None
    seed: int | None = None
    rng: np.random.Generator = field(init=False)

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.seed)

    # state as tensor with one axis per live qubit
    def run(self, program: Program, callbacks: Callbacks | None = None) -> list[int]:
        """Run once.  Returns the final measurement stack (bottom -> top)."""
        self._state = np.ones((), dtype=complex)
        self._axes: dict[Qubit, int] = {}
        self._callbacks = callbacks or Callbacks()
        self._gateset = program.gateset
        mstack = MeasurementStack()
        self._kernel(program.kernel, mstack)
        return list(mstack.bits)

    def sample(self, program: Program, callbacks: Callbacks | None = None, shots: int = 1000) -> dict[tuple[int, ...], int]:
        counts: dict[tuple[int, ...], int] = {}
        for _ in range(shots):
            k = tuple(self.run(program, callbacks))
            counts[k] = counts.get(k, 0) + 1
        return counts

    # ---- internals -------------------------------------------------------------
    def _kernel(self, k: Kernel, mstack: MeasurementStack) -> None:
        if k.target is not None:
            self._allocate(k.target)
        for ins in k.body:
            if isinstance(ins, Gate):
                self._gate(ins)
            else:
                self._kernel(ins, mstack)
        if k.target is None:
            return
        mstack.push(self._measure(k.target))
        fn = self._callbacks[k.callback.name]
        new = fn(mstack, **k.callback.params)
        if new is not None:
            self._gateset.check(new)
            self._kernel(new, mstack)

    def _allocate(self, q: Qubit) -> None:
        if q in self._axes:
            raise RuntimeError(f"qubit {q} already allocated")
        self._state = np.tensordot(self._state, np.array([1, 0], dtype=complex), axes=0)
        self._axes[q] = self._state.ndim - 1

    def _apply(self, u: np.ndarray, qubits: tuple[Qubit, ...]) -> None:
        k = len(qubits)
        axes = [self._axes[q] for q in qubits]
        u = u.reshape((2,) * (2 * k))
        # contract u's input legs (last k) with the state's chosen axes
        st = np.tensordot(u, self._state, axes=(list(range(k, 2 * k)), axes))
        # tensordot puts the k new output legs first; move them back
        st = np.moveaxis(st, list(range(k)), axes)
        self._state = st

    def _gate(self, g: Gate) -> None:
        self._apply(unitary(g), g.qubits)
        if self.noise is not None:
            for q in g.qubits:
                e = self.noise.sample(self.rng)
                if e is not None:
                    self._apply(e, (q,))

    def _measure(self, q: Qubit) -> int:
        ax = self._axes[q]
        probs = np.sum(np.abs(np.moveaxis(self._state, ax, 0)) ** 2, axis=tuple(range(1, self._state.ndim)))
        p1 = float(probs[1])
        bit = int(self.rng.random() < p1)
        st = np.take(self._state, bit, axis=ax)
        norm = np.linalg.norm(st)
        self._state = st / norm
        del self._axes[q]
        for other, a in self._axes.items():
            if a > ax:
                self._axes[other] = a - 1
        return bit
