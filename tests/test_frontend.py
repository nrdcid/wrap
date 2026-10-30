import pytest

from wrap.frontend.toy import allocate, entangle, measure, mix, to_program, toy
from wrap.ir import CallbackRef, Qubit


def test_lowers_to_nested_kernels():
    @toy
    def prog():
        q1 = allocate()
        q2 = allocate()
        mix(q2)
        entangle(q2, q1)
        measure(q2, callback="fix", params=(q1,))
        measure(q1)

    p = to_program(prog)
    k = p.kernel
    assert k.target == Qubit("q1")
    inner = k.body[0]
    assert inner.target == Qubit("q2")
    assert inner.callback == CallbackRef("fix", {"q0": Qubit("q1")})
    assert [g.name for g in inner.body] == ["mix", "entangle"]


def test_lifo_violation_is_rejected():
    @toy
    def prog():
        q1 = allocate()
        q2 = allocate()
        measure(q1)
        measure(q2)

    with pytest.raises(ValueError, match="LIFO"):
        to_program(prog)


def test_unmeasured_qubit_is_rejected():
    @toy
    def prog():
        q1 = allocate()
        mix(q1)

    with pytest.raises(ValueError, match="never measured"):
        to_program(prog)
