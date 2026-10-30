from dataclasses import dataclass, field
from typing import ClassVar

import pytest

from wrap import CLIFFORD, TOY, Callbacks, Compiler, Gate, GateSet, Kernel, Pipeline, Program, Qubit
from wrap.compiler import UnsupportedGate
from wrap.mstack import Decoder, majority
from wrap.passes import BitFlipRep3, CliffordToH2, PhaseFlipRep3, ToyToClifford


def test_pipeline_rejects_gateset_mismatch():
    with pytest.raises(ValueError, match="cannot chain"):
        Pipeline([ToyToClifford(), ToyToClifford()])


def test_pass_rejects_wrong_source():
    prog = Program(CLIFFORD, Kernel([Gate("X", (Qubit("q"),))], target=Qubit("q")))
    with pytest.raises(ValueError, match="expected program in gate set"):
        ToyToClifford().compile(prog)


def test_decoder_arity_must_match_n_physical():
    @dataclass
    class Bad(Compiler):
        source: ClassVar[GateSet] = CLIFFORD
        target: ClassVar[GateSet] = CLIFFORD
        name: ClassVar[str] = "bad"
        n_physical: ClassVar[int] = 3
        decoder: Decoder = field(default_factory=lambda: Decoder(2, 1, majority))

    with pytest.raises(ValueError, match="arity"):
        Bad()


def test_unsupported_gate_raises():
    prog = Program(CLIFFORD, Kernel([Gate("CZ", (Qubit("a"), Qubit("b")))], target=Qubit("a")))
    # CZ is defined for rep3-bit; make a toy pass that lacks it
    @dataclass
    class NoCZ(Compiler):
        source: ClassVar[GateSet] = CLIFFORD
        target: ClassVar[GateSet] = CLIFFORD
        name: ClassVar[str] = "nocz"

    with pytest.raises(UnsupportedGate):
        NoCZ().compile(prog)


def test_physical_qubit_count_multiplies():
    q = Qubit("q")
    prog = Program(TOY, Kernel([Gate("flip", (q,))], target=q))
    pipe = Pipeline([ToyToClifford(), BitFlipRep3(), PhaseFlipRep3()])
    out, cbs = pipe.compile(prog)
    assert pipe.physical_per_logical == 9
    assert len({k.target for k in out.kernel.kernels()}) == 9
    # each pass contributes one distinct no-op callback; the user's `done` survives
    assert {"done", "__noop@toy->clifford", "__noop@rep3-bit", "__noop@rep3-phase"} == set(cbs)


def test_returned_kernels_are_compiled_through_the_pipeline():
    """The paper's key property: a callback written against TOY returns a TOY
    kernel; after compilation the wrapper lowers it to the target ISA."""
    q = Qubit("q")

    def fix(mstack, *, q0):
        return Kernel([Gate("flip", (q0,))])

    prog = Program(TOY, Kernel([], target=q, callback=__import__("wrap").CallbackRef("fix", {"q0": q})))
    out, cbs = Pipeline([ToyToClifford(), BitFlipRep3(), CliffordToH2()]).compile(prog, Callbacks().with_(fix=fix))

    from wrap.mstack import MeasurementStack

    ms = MeasurementStack([0, 1, 1])  # three physical bits for the one logical measurement
    k = cbs["fix"](ms, q0=q)
    assert ms.bits == [1]  # majority-decoded to one logical bit
    names = [g.name for g in k.gates()]
    assert set(names) <= {"RZ", "U1q", "ZZ"}  # TOY flip -> X -> XXX -> U1q x3
    assert len(names) == 3
