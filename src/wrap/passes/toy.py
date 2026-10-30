"""ISA pass: toy -> clifford."""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..compiler import Compiler, handles
from ..ir import CLIFFORD, TOY, Gate, GateSet, Instruction


@dataclass
class ToyToClifford(Compiler):
    source: ClassVar[GateSet] = TOY
    target: ClassVar[GateSet] = CLIFFORD
    name: ClassVar[str] = "toy->clifford"

    @handles("flip")
    def flip(self, g: Gate) -> list[Instruction]:
        return [Gate("X", g.qubits)]

    @handles("mix")
    def mix(self, g: Gate) -> list[Instruction]:
        return [Gate("H", g.qubits)]

    @handles("entangle")
    def entangle(self, g: Gate) -> list[Instruction]:
        return [Gate("CX", g.qubits)]
