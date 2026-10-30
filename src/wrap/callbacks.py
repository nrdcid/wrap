"""Callbacks: opaque classical functions keyed by name.

Signature: ``cb(mstack, **params) -> Kernel | None``.
Returning a kernel executes it immediately (in the callback's *source* gate
set; the framework compiles it on the fly).  Returning None ends the current
kernel.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Iterator, Mapping, Optional

from .ir import Kernel
from .mstack import MeasurementStack

CallbackFn = Callable[..., Optional[Kernel]]


def done(mstack: MeasurementStack, **_: Any) -> None:
    """Default callback: leave the stack intact, return no kernel."""
    return None


@dataclass
class Callbacks(Mapping[str, CallbackFn]):
    """An immutable-by-convention name -> function table."""

    table: dict[str, CallbackFn] = field(default_factory=dict)

    def __post_init__(self) -> None:
        self.table.setdefault("done", done)

    def __getitem__(self, name: str) -> CallbackFn:
        try:
            return self.table[name]
        except KeyError:
            raise KeyError(f"unknown callback {name!r}; known: {sorted(self.table)}") from None

    def __iter__(self) -> Iterator[str]:
        return iter(self.table)

    def __len__(self) -> int:
        return len(self.table)

    def with_(self, **fns: CallbackFn) -> "Callbacks":
        return Callbacks({**self.table, **fns})

    def map(self, f: Callable[[str, CallbackFn], CallbackFn]) -> "Callbacks":
        return Callbacks({name: f(name, fn) for name, fn in self.table.items()})


def callback(fn: CallbackFn) -> CallbackFn:
    """Decorator marker (purely cosmetic for now; reserved for static checks)."""
    fn.__wrap_callback__ = True  # type: ignore[attr-defined]
    return fn
