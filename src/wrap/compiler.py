"""Compiler passes and callback wrapping.

A pass is a function Program -> Program between gate sets.  Two kinds:

* **ISA passes** change the gate set (toy -> clifford -> h2) and leave the
  measurement structure alone (`decoder = IDENTITY`).
* **QEC passes** preserve the gate set (clifford -> clifford) but expand each
  logical qubit into `n` physical qubits and supply a `decoder` with arity
  n -> 1.

`Compiler.compile` returns *both* the transformed program and a transformed
callback table.  Every callback is wrapped so that

    wrapped(mstack) = decoder(mstack); k = original(mstack); compile(k)

Chaining passes is function composition; no pass knows about any other.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, ClassVar

from .callbacks import CallbackFn, Callbacks, done
from .ir import CallbackRef, Gate, GateSet, Instruction, Kernel, Program, Qubit
from .mstack import IDENTITY, Decoder, MeasurementStack

Handler = Callable[["Compiler", Gate], list[Instruction]]


class UnsupportedGate(NotImplementedError):
    pass


def handles(*names: str) -> Callable[[Handler], Handler]:
    """Register a method as the lowering for gate `name`."""

    def deco(fn: Handler) -> Handler:
        fn.__wrap_handles__ = names  # type: ignore[attr-defined]
        return fn

    return deco


@dataclass
class Compiler:
    """Base class.  Subclasses set `source`, `target`, optionally `decoder`,
    `n_physical`, `prepare`, `pre_measure`, and `@handles(...)` methods."""

    source: ClassVar[GateSet]
    target: ClassVar[GateSet]
    name: ClassVar[str] = "pass"

    #: physical qubits per logical qubit (1 for ISA passes)
    n_physical: ClassVar[int] = 1
    #: measurement-stack transformer; arity must be (n_physical -> 1)
    decoder: Decoder = field(default_factory=lambda: IDENTITY)

    _handlers: dict[str, Handler] = field(default_factory=dict, init=False, repr=False)

    def __post_init__(self) -> None:
        for attr in dir(type(self)):
            fn = getattr(type(self), attr)
            for g in getattr(fn, "__wrap_handles__", ()):
                self._handlers[g] = fn
        if self.n_physical > 1 and (self.decoder.in_bits, self.decoder.out_bits) != (
            self.n_physical,
            1,
        ):
            raise ValueError(
                f"{self.name}: decoder arity {self.decoder.in_bits}->{self.decoder.out_bits} "
                f"does not match n_physical={self.n_physical}"
            )

    # ---- hooks subclasses may override ----------------------------------------
    def encode(self, q: Qubit) -> list[Qubit]:
        """Physical qubits realising logical qubit `q`."""
        if self.n_physical == 1:
            return [q]
        return [q.sub(i) for i in range(self.n_physical)]

    def prepare(self, q: Qubit) -> list[Instruction]:
        """Gates run right after allocation to prepare the logical |0>."""
        return []

    def pre_measure(self, q: Qubit) -> list[Instruction]:
        """Gates run right before the physical measurements (basis change)."""
        return []

    @property
    def noop(self) -> CallbackRef:
        """A pass-unique no-op callback used on inner physical kernels.

        It must be distinct from the user's `done`: later passes wrap *all*
        callbacks in the table, including this one, which is exactly what
        makes code concatenation work."""
        return CallbackRef(f"__noop@{self.name}")

    @property
    def compiler_callbacks(self) -> dict[str, CallbackFn]:
        return {self.noop.name: done}

    # ---- gate lowering -----------------------------------------------------------
    def compile_gate(self, gate: Gate) -> list[Instruction]:
        try:
            handler = self._handlers[gate.name]
        except KeyError:
            raise UnsupportedGate(f"{self.name}: no handler for gate {gate.name!r}") from None
        return handler(self, gate)

    # ---- kernel lowering ---------------------------------------------------------
    def compile_kernel(self, kernel: Kernel) -> Kernel:
        body: list[Instruction] = []
        if kernel.target is not None:
            body += self.prepare(kernel.target)
        for ins in kernel.body:
            if isinstance(ins, Gate):
                body += self.compile_gate(ins)
            else:
                body.append(self.compile_kernel(ins))
        if kernel.target is None:
            return Kernel(body=body)
        body += self.pre_measure(kernel.target)

        # Nest physical allocations: q.(n-1) innermost (measured first), q.0
        # outermost carrying the original callback, so that all n bits are on
        # the stack when the (wrapped) callback runs.
        phys = self.encode(kernel.target)
        inner = Kernel(body=body, target=phys[-1], callback=self.noop)
        for pq in reversed(phys[:-1]):
            inner = Kernel(body=[inner], target=pq, callback=self.noop)
        inner.callback = kernel.callback
        return inner

    # ---- callback wrapping -------------------------------------------------------
    def wrap(self, original: CallbackFn) -> CallbackFn:
        def wrapped(mstack: MeasurementStack, **params: Any) -> Kernel | None:
            self.decoder(mstack)  # restore the stack to what `original` expects
            k = original(mstack, **params)
            if k is None:
                return None
            self.source.check(k)
            return self.compile_kernel(k)  # returned kernels go through this pass too

        wrapped.__name__ = f"{self.name}<{getattr(original, '__name__', 'cb')}>"
        return wrapped

    def wrap_callbacks(self, callbacks: Callbacks) -> Callbacks:
        return callbacks.map(lambda _name, fn: self.wrap(fn))

    # ---- entry point ----------------------------------------------------------------
    def compile(self, program: Program, callbacks: Callbacks | None = None) -> tuple[Program, Callbacks]:
        if program.gateset != self.source:
            raise ValueError(
                f"{self.name}: expected program in gate set {self.source.name!r}, got {program.gateset.name!r}"
            )
        new_callbacks = self.wrap_callbacks(callbacks or Callbacks())
        new_callbacks = new_callbacks.with_(**self.compiler_callbacks)
        new_kernel = self.compile_kernel(program.kernel)
        return Program(self.target, new_kernel), new_callbacks


@dataclass
class Pipeline:
    """A list of passes applied in order.  Checks gate-set compatibility."""

    passes: list[Compiler]

    def __post_init__(self) -> None:
        for a, b in zip(self.passes, self.passes[1:]):
            if a.target != b.source:
                raise ValueError(f"cannot chain {a.name} ({a.target.name}) -> {b.name} ({b.source.name})")

    def compile(self, program: Program, callbacks: Callbacks | None = None) -> tuple[Program, Callbacks]:
        cbs = callbacks or Callbacks()
        for p in self.passes:
            program, cbs = p.compile(program, cbs)
        return program, cbs

    def stages(self, program: Program, callbacks: Callbacks | None = None):
        """Yield (pass_name, program, callbacks) after each pass; stage 0 is the input."""
        cbs = callbacks or Callbacks()
        yield "source", program, cbs
        for p in self.passes:
            program, cbs = p.compile(program, cbs)
            yield p.name, program, cbs

    @property
    def physical_per_logical(self) -> int:
        n = 1
        for p in self.passes:
            n *= p.n_physical
        return n
