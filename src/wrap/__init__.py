"""wrap — QEC is just a compiler pass.

An open fault-tolerant compiler stack built on *callback wrapping*
(Paz & Grossman, arXiv:2605.16595).
"""

from .ir import CLIFFORD, H2, TOY, CallbackRef, Gate, GateSet, Kernel, Program, Qubit
from .callbacks import Callbacks, callback, done
from .compiler import Compiler, Pipeline, UnsupportedGate, handles
from .mstack import Decoder, MeasurementStack, majority, parity

__all__ = [
    "CLIFFORD", "H2", "TOY", "CallbackRef", "Gate", "GateSet", "Kernel", "Program", "Qubit",
    "Callbacks", "callback", "done",
    "Compiler", "Pipeline", "UnsupportedGate", "handles",
    "Decoder", "MeasurementStack", "majority", "parity",
]
__version__ = "0.0.1"
