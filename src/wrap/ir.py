"""Kernel IR: the purely quantum intermediate representation.

Model (after Paz & Grossman, arXiv:2605.16595): a program is a tree of
*kernels*.  A kernel optionally allocates one qubit, runs a list of
instructions (gates or nested kernels), measures its qubit, pushes the
outcome on the measurement stack, and invokes an opaque classical
*callback*.  The callback may return a new kernel (in the *source* gate
set) that is executed immediately in the enclosing scope.

Classical logic never appears in the IR; only callback names and constant
parameters do.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Union


@dataclass(frozen=True, order=True)
class Qubit:
    """A qubit identifier.  QEC passes refine `q` into `q.0`, `q.1`, ..."""

    name: str

    def sub(self, i: int) -> "Qubit":
        return Qubit(f"{self.name}.{i}")

    def __str__(self) -> str:
        return self.name


@dataclass(frozen=True)
class Gate:
    name: str
    qubits: tuple[Qubit, ...]
    params: tuple[Any, ...] = ()

    def __str__(self) -> str:
        qs = " ".join(map(str, self.qubits))
        ps = f"({', '.join(map(str, self.params))})" if self.params else ""
        return f"{self.name}{ps} {qs}"


@dataclass(frozen=True)
class CallbackRef:
    """Reference to an opaque classical callback plus constant parameters.

    Parameters are visible to the compiler (e.g. a qubit to fix up); the body
    is not.  `done` is the default no-op callback.
    """

    name: str = "done"
    params: Mapping[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if not self.params:
            return self.name
        ps = ", ".join(f"{k}={v}" for k, v in self.params.items())
        return f"{self.name}({ps})"

    def __hash__(self) -> int:
        return hash((self.name, tuple(sorted(self.params.items(), key=lambda kv: kv[0]))))


Instruction = Union[Gate, "Kernel"]


@dataclass
class Kernel:
    """allocate `target`: run `body`; measure `target`; call `callback`.

    A kernel with `target=None` is a pure compute block: no allocation, no
    measurement, no callback (used for callback-returned fixups).
    """

    body: list[Instruction] = field(default_factory=list)
    target: Qubit | None = None
    callback: CallbackRef = field(default_factory=CallbackRef)

    # ---- structural helpers -------------------------------------------------
    def gates(self) -> Iterator[Gate]:
        for ins in self.body:
            if isinstance(ins, Gate):
                yield ins
            else:
                yield from ins.gates()

    def kernels(self) -> Iterator["Kernel"]:
        yield self
        for ins in self.body:
            if isinstance(ins, Kernel):
                yield from ins.kernels()

    def callbacks(self) -> set[str]:
        return {k.callback.name for k in self.kernels() if k.target is not None}

    def qubits(self) -> set[Qubit]:
        qs = {k.target for k in self.kernels() if k.target is not None}
        for g in self.gates():
            qs.update(g.qubits)
        return qs

    def pretty(self, indent: int = 0) -> str:
        pad = "  " * indent
        lines: list[str] = []
        if self.target is not None:
            lines.append(f"{pad}allocate {self.target}:")
            inner = indent + 1
        else:
            inner = indent
        for ins in self.body:
            if isinstance(ins, Gate):
                lines.append("  " * inner + str(ins))
            else:
                lines.append(ins.pretty(inner))
        if self.target is not None:
            lines.append(f"{pad}measure {self.callback}")
        return "\n".join(lines)

    def __str__(self) -> str:
        return self.pretty()


@dataclass(frozen=True)
class GateSet:
    """An instruction set: gate name -> arity."""

    name: str
    gates: Mapping[str, int]

    def __contains__(self, gate: str) -> bool:
        return gate in self.gates

    def check(self, kernel: Kernel) -> None:
        for g in kernel.gates():
            if g.name not in self.gates:
                raise ValueError(f"gate {g.name!r} not in gate set {self.name!r}")
            if len(g.qubits) != self.gates[g.name]:
                raise ValueError(
                    f"gate {g.name!r} expects {self.gates[g.name]} qubits, got {len(g.qubits)}"
                )


@dataclass
class Program:
    """A kernel tagged with the gate set it is expressed in."""

    gateset: GateSet
    kernel: Kernel

    def __post_init__(self) -> None:
        self.gateset.check(self.kernel)

    def __str__(self) -> str:
        return f"# gateset: {self.gateset.name}\n{self.kernel}"


# ---- standard gate sets ------------------------------------------------------

TOY = GateSet("toy", {"flip": 1, "mix": 1, "entangle": 2})
"""Pedagogical source ISA: flip=X, mix=H, entangle=CX."""

CLIFFORD = GateSet("clifford", {"X": 1, "Y": 1, "Z": 1, "H": 1, "S": 1, "CX": 2, "CZ": 2})
"""The framework's standard intermediate ISA.  QEC passes map CLIFFORD -> CLIFFORD."""

H2 = GateSet("h2", {"RZ": 1, "U1q": 1, "ZZ": 2})
"""Trapped-ion native ISA (Quantinuum H-series style).  Params: angles."""
