import pytest

from wrap.mstack import IDENTITY, Decoder, MeasurementStack, StackUnderflow, majority, parity


def test_identity_does_not_drain_stack():
    ms = MeasurementStack([1, 0, 1])
    IDENTITY(ms)
    assert ms.bits == [1, 0, 1]


def test_pop_n_order_and_underflow():
    ms = MeasurementStack([1, 0, 1, 1])
    assert ms.pop_n(3) == [0, 1, 1]
    assert ms.bits == [1]
    with pytest.raises(StackUnderflow):
        ms.pop_n(2)


def test_majority_decoder_arity_checked():
    d = Decoder(3, 1, majority)
    ms = MeasurementStack([0, 1, 0, 1])
    d(ms)
    assert ms.bits == [0, 1]
    bad = Decoder(2, 1, lambda b: b)  # returns 2 bits, claims 1
    with pytest.raises(ValueError):
        bad(MeasurementStack([0, 1]))


def test_parity():
    assert parity([1, 1, 0]) == [0]
    assert parity([1, 0, 0]) == [1]
