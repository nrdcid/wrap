"""QEC passes: distance-3 repetition codes (clifford -> clifford).

Composing BitFlipRep3 then PhaseFlipRep3 yields a 9-qubit Shor-type code
(bit-flip outer, phase-flip inner) that corrects one X and one Z error.
Composing the same pass twice yields a distance-9 repetition code.

Neither code has a transversal H.  `H` (and `S` on the phase-flip code) are
implemented by *unencode -> apply -> re-encode*, which is NOT fault tolerant:
it exposes the logical qubit on one physical qubit for the duration of the
gate.  It keeps pipelines total for demos; a real FT H needs a different code
or code switching (see roadmap).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar

from ..compiler import Compiler, handles
from ..ir import CLIFFORD, Gate, GateSet, Instruction, Qubit
from ..mstack import Decoder, majority


def _majority_decoder(name: str) -> Decoder:
    return Decoder(3, 1, majority, name=name)


@dataclass
class BitFlipRep3(Compiler):
    """|0_L> = |000>.  Corrects a single X error via majority vote."""

    source: ClassVar[GateSet] = CLIFFORD
    target: ClassVar[GateSet] = CLIFFORD
    name: ClassVar[str] = "rep3-bit"
    n_physical: ClassVar[int] = 3
    decoder: Decoder = field(default_factory=lambda: _majority_decoder("rep3-bit"))

    @handles("X")
    def x(self, g: Gate) -> list[Instruction]:
        return [Gate("X", (p,)) for p in self.encode(g.qubits[0])]

    @handles("Y")
    def y(self, g: Gate) -> list[Instruction]:
        # Y_L = i X_L Z_L  (global phase irrelevant)
        return self.x(g) + self.z(g)

    @handles("Z")
    def z(self, g: Gate) -> list[Instruction]:
        return [Gate("Z", (p,)) for p in self.encode(g.qubits[0])]

    @handles("S")
    def s(self, g: Gate) -> list[Instruction]:
        # S_L = diag(1, i) on {|000>,|111>}: S on one qubit suffices.
        return [Gate("S", (self.encode(g.qubits[0])[0],))]

    def unencoded(self, q: Qubit, gates: list[Instruction]) -> list[Instruction]:
        """Non-FT escape hatch: decode onto q.0, apply `gates`, re-encode."""
        p = self.encode(q)
        enc = [Gate("CX", (p[0], p[1])), Gate("CX", (p[0], p[2]))]
        return enc + gates + enc

    @handles("H")
    def h(self, g: Gate) -> list[Instruction]:
        q = g.qubits[0]
        return self.unencoded(q, [Gate("H", (self.encode(q)[0],))])

    @handles("CX")
    def cx(self, g: Gate) -> list[Instruction]:
        c, t = (self.encode(q) for q in g.qubits)
        return [Gate("CX", (ci, ti)) for ci, ti in zip(c, t)]

    @handles("CZ")
    def cz(self, g: Gate) -> list[Instruction]:
        c, t = (self.encode(q) for q in g.qubits)
        # Z_L on target controlled by logical control: CZ between one pair suffices.
        return [Gate("CZ", (c[0], t[0]))]


@dataclass
class PhaseFlipRep3(Compiler):
    """|0_L> = |+++>.  Corrects a single Z error (X-basis majority vote).

    Logical operators are the Hadamard conjugates of the bit-flip code:
    X_L = ZZZ, Z_L = XXX, and transversal CX implements CX with control and
    target *swapped*.
    """

    source: ClassVar[GateSet] = CLIFFORD
    target: ClassVar[GateSet] = CLIFFORD
    name: ClassVar[str] = "rep3-phase"
    n_physical: ClassVar[int] = 3
    decoder: Decoder = field(default_factory=lambda: _majority_decoder("rep3-phase"))

    def prepare(self, q: Qubit) -> list[Instruction]:
        return [Gate("H", (p,)) for p in self.encode(q)]

    def pre_measure(self, q: Qubit) -> list[Instruction]:
        return [Gate("H", (p,)) for p in self.encode(q)]

    @handles("X")
    def x(self, g: Gate) -> list[Instruction]:
        return [Gate("Z", (p,)) for p in self.encode(g.qubits[0])]

    @handles("Z")
    def z(self, g: Gate) -> list[Instruction]:
        return [Gate("X", (p,)) for p in self.encode(g.qubits[0])]

    @handles("Y")
    def y(self, g: Gate) -> list[Instruction]:
        return self.x(g) + self.z(g)

    @handles("CX")
    def cx(self, g: Gate) -> list[Instruction]:
        c, t = (self.encode(q) for q in g.qubits)
        return [Gate("CX", (ti, ci)) for ci, ti in zip(c, t)]

    def unencoded(self, q: Qubit, gates: list[Instruction]) -> list[Instruction]:
        """Non-FT escape hatch: H^3 maps this code onto the bit-flip code."""
        p = self.encode(q)
        hs = [Gate("H", (x,)) for x in p]
        enc = [Gate("CX", (p[0], p[1])), Gate("CX", (p[0], p[2]))]
        return hs + enc + gates + enc + hs

    @handles("H")
    def h(self, g: Gate) -> list[Instruction]:
        q = g.qubits[0]
        return self.unencoded(q, [Gate("H", (self.encode(q)[0],))])

    @handles("S")
    def s(self, g: Gate) -> list[Instruction]:
        q = g.qubits[0]
        return self.unencoded(q, [Gate("S", (self.encode(q)[0],))])

    @handles("CZ")
    def cz(self, g: Gate) -> list[Instruction]:
        # CZ = (I x H) CX (I x H); with the swapped-CX rule above this is the
        # cleanest correct form.  Non-FT because of the H.
        c, t = g.qubits
        return self.h(Gate("H", (t,))) + self.cx(Gate("CX", (c, t))) + self.h(Gate("H", (t,)))
