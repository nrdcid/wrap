"""Surface-code QEC pass — DESIGN STUB (not implemented).

This is the open question the qstack paper leaves: does callback wrapping
survive codes whose decoder is an iterative algorithm (MWPM) rather than a
finite lookup?  Plan:

* `n_physical = d*d` data qubits + `d*d - 1` ancillas per logical qubit;
  allocation nests the data qubits (LIFO measurement) and ancillas as
  compute-only sub-kernels that are measured round by round.
* Logical gates: transversal X/Z, H via patch rotation (or non-FT fallback),
  CX via lattice surgery **as a nested pass** (re-use the LS compiler
  literature as plug-ins rather than competitors).
* `decoder`: a `Decoder(in_bits=d*d + rounds*(d*d-1), out_bits=1)` whose `fn`
  builds a `pymatching.Matching` from the code's detector graph and returns
  the corrected logical measurement.  Because the decoder sees *all* the
  physical bits of its own kernel on top of the stack, wrapping composes with
  outer codes exactly as for rep-3.
* Stim: emit DETECTOR annotations so `circuit.detector_error_model()` feeds
  PyMatching directly on the static path.

Good first issue once the stub is filled: distance-3 rotated surface code,
memory experiment, compare logical error vs. BitFlipRep3 at p=1e-3.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar

from ..compiler import Compiler
from ..ir import CLIFFORD, GateSet


@dataclass
class SurfaceCode(Compiler):
    source: ClassVar[GateSet] = CLIFFORD
    target: ClassVar[GateSet] = CLIFFORD
    name: ClassVar[str] = "surface"
    distance: int = 3

    def __post_init__(self) -> None:
        raise NotImplementedError("SurfaceCode pass is a design stub; see module docstring.")
