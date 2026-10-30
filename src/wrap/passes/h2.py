"""ISA pass: clifford -> h2 (trapped-ion native: RZ, U1q, ZZ).

Conventions (Quantinuum-style):
    RZ(t)      = exp(-i t Z / 2)
    U1q(t, p)  = exp(-i t (cos p X + sin p Y) / 2)
    ZZ         = exp(-i pi Z⊗Z / 4)
Decompositions are verified up to global phase in tests/test_h2.py.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import pi
from typing import ClassVar

from ..compiler import Compiler, handles
from ..ir import CLIFFORD, H2, Gate, GateSet, Instruction


@dataclass
class CliffordToH2(Compiler):
    source: ClassVar[GateSet] = CLIFFORD
    target: ClassVar[GateSet] = H2
    name: ClassVar[str] = "clifford->h2"

    @handles("X")
    def x(self, g: Gate) -> list[Instruction]:
        return [Gate("U1q", g.qubits, (pi, 0.0))]

    @handles("Y")
    def y(self, g: Gate) -> list[Instruction]:
        return [Gate("U1q", g.qubits, (pi, pi / 2))]

    @handles("Z")
    def z(self, g: Gate) -> list[Instruction]:
        return [Gate("RZ", g.qubits, (pi,))]

    @handles("S")
    def s(self, g: Gate) -> list[Instruction]:
        return [Gate("RZ", g.qubits, (pi / 2,))]

    @handles("H")
    def h(self, g: Gate) -> list[Instruction]:
        return [Gate("U1q", g.qubits, (pi / 2, -pi / 2)), Gate("RZ", g.qubits, (pi,))]

    @handles("CZ")
    def cz(self, g: Gate) -> list[Instruction]:
        c, t = g.qubits
        return [Gate("ZZ", (c, t)), Gate("RZ", (c,), (-pi / 2,)), Gate("RZ", (t,), (-pi / 2,))]

    @handles("CX")
    def cx(self, g: Gate) -> list[Instruction]:
        c, t = g.qubits
        return [
            Gate("U1q", (t,), (-pi / 2, pi / 2)),
            Gate("ZZ", (c, t)),
            Gate("RZ", (c,), (-pi / 2,)),
            Gate("U1q", (t,), (pi / 2, pi)),
            Gate("RZ", (t,), (-pi / 2,)),
        ]
