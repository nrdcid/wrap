"""Semantic preservation across compilation stages.

Two flavours:
* exact: programs built from classical-reversible gates (flip/entangle) plus
  bit-driven fixup callbacks are deterministic -> every stage must agree bit
  for bit;
* statistical (Hypothesis): random toy programs incl. `mix`; compare stage
  distributions by total-variation distance on the Stim backend.
"""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from wrap import CallbackRef, Callbacks, Gate, Kernel, Pipeline, Program, Qubit, TOY
from wrap.backends import Emulator, StimBackend
from wrap.passes import BitFlipRep3, CliffordToH2, PhaseFlipRep3, ToyToClifford

QS = [Qubit("a"), Qubit("b"), Qubit("c")]


def flip_if_one(mstack, *, q0):
    return Kernel([Gate("flip", (q0,))]) if mstack.bits[-1] else None


def flip_if_zero(mstack, *, q0):
    return Kernel([Gate("flip", (q0,))]) if not mstack.bits[-1] else None


CBS = Callbacks().with_(flip_if_one=flip_if_one, flip_if_zero=flip_if_zero)


# ---- program generator ----------------------------------------------------------
@st.composite
def toy_programs(draw, n_qubits: int, allow_mix: bool):
    """Nested kernels a ⊃ b ⊃ c with random gates and callbacks."""
    live = QS[:n_qubits]
    gate_names = ["flip", "entangle"] + (["mix"] if allow_mix else [])

    def gates(available):
        out = []
        for _ in range(draw(st.integers(0, 4))):
            g = draw(st.sampled_from(gate_names))
            if g == "entangle" and len(available) >= 2:
                c, t = draw(st.permutations(available).map(lambda p: p[:2]))
                out.append(Gate("entangle", (c, t)))
            elif g != "entangle":
                out.append(Gate(g, (draw(st.sampled_from(available)),)))
        return out

    def callback(outer):
        if not outer or draw(st.booleans()):
            return CallbackRef()
        name = draw(st.sampled_from(["flip_if_one", "flip_if_zero"]))
        return CallbackRef(name, {"q0": draw(st.sampled_from(outer))})

    kernel = None
    for depth in reversed(range(n_qubits)):
        q = live[depth]
        avail = live[: depth + 1]
        body = gates(avail)
        if kernel is not None:
            body.insert(draw(st.integers(0, len(body))), kernel)
            body += gates(avail)
        kernel = Kernel(body, target=q, callback=callback(live[:depth]))
    return Program(TOY, kernel)


def _tv(p, q):
    keys = set(p) | set(q)
    np_, nq = sum(p.values()), sum(q.values())
    return 0.5 * sum(abs(p.get(k, 0) / np_ - q.get(k, 0) / nq) for k in keys)


PIPE = Pipeline([ToyToClifford(), BitFlipRep3(), PhaseFlipRep3()])
H2PIPE = Pipeline([ToyToClifford(), BitFlipRep3(), CliffordToH2()])


@settings(max_examples=40, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(toy_programs(n_qubits=3, allow_mix=False))
def test_exact_deterministic_programs_agree_at_every_stage(program):
    emu = Emulator(seed=0)
    stim = StimBackend()
    expected = emu.run(program, CBS)
    for name, prog, cbs in PIPE.stages(program, CBS):
        backend = emu if prog.gateset == TOY else stim
        got = backend.run(prog, cbs) if backend is emu else stim.run(prog, cbs, seed=1)
        assert got == expected, f"stage {name}: {got} != {expected}"
    prog, cbs = H2PIPE.compile(program, CBS)
    assert emu.run(prog, cbs) == expected


@settings(max_examples=15, deadline=None, suppress_health_check=[HealthCheck.too_slow])
@given(toy_programs(n_qubits=2, allow_mix=True))
def test_statistical_agreement_with_mix(program):
    emu = Emulator(seed=0)
    stim = StimBackend()
    ref = emu.sample(program, CBS, shots=600)
    for name, prog, cbs in PIPE.stages(program, CBS):
        if prog.gateset == TOY:
            continue
        got = stim.sample(prog, cbs, shots=3000, seed=2)
        assert _tv(ref, got) < 0.12, f"stage {name}: TV={_tv(ref, got):.3f}"
