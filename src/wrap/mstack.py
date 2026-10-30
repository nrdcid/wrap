"""The measurement stack and typed stack transformers.

Every kernel pushes its measurement outcome when it completes.  Because
inner kernels complete before outer ones (LIFO), a QEC pass that expands one
logical qubit into n physical qubits is guaranteed that the n physical bits
sit on top of the stack when the wrapped callback runs.

`StackTransformer` is the generalisation of the paper's `decode` method:
a function on the stack with declared arity (in_bits -> out_bits).  Making the
arity explicit lets the framework *check* wrapping order instead of relying on
the LIFO convention silently.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Protocol


class StackUnderflow(RuntimeError):
    pass


@dataclass
class MeasurementStack:
    bits: list[int] = field(default_factory=list)

    def push(self, bit: int) -> None:
        self.bits.append(int(bit) & 1)

    def pop(self) -> int:
        if not self.bits:
            raise StackUnderflow("measurement stack underflow")
        return self.bits.pop()

    def pop_n(self, n: int) -> list[int]:
        """Pop n bits; returned list is in push order (oldest first)."""
        if n == 0:
            return []
        if len(self.bits) < n:
            raise StackUnderflow(f"need {n} bits, have {len(self.bits)}")
        out = self.bits[-n:]
        del self.bits[-n:]
        return out

    def __len__(self) -> int:
        return len(self.bits)

    def copy(self) -> "MeasurementStack":
        return MeasurementStack(list(self.bits))


class StackTransformer(Protocol):
    """Consumes `in_bits` from the top of the stack, pushes `out_bits`."""

    in_bits: int
    out_bits: int

    def __call__(self, mstack: MeasurementStack) -> None: ...


@dataclass
class Decoder:
    """A StackTransformer built from a plain function on bit lists."""

    in_bits: int
    out_bits: int
    fn: Callable[[list[int]], list[int]]
    name: str = "decoder"

    def __call__(self, mstack: MeasurementStack) -> None:
        raw = mstack.pop_n(self.in_bits)
        out = self.fn(raw)
        if len(out) != self.out_bits:
            raise ValueError(f"{self.name}: expected {self.out_bits} output bits, got {len(out)}")
        for b in out:
            mstack.push(b)


IDENTITY = Decoder(0, 0, lambda bits: [], name="identity")


def majority(bits: list[int]) -> list[int]:
    return [1 if sum(bits) * 2 > len(bits) else 0]


def parity(bits: list[int]) -> list[int]:
    return [sum(bits) & 1]
