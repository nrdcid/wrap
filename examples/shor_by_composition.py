"""The paper's headline demo: a program with a classical fixup callback is
compiled toy -> clifford -> rep3-bit -> rep3-phase (= 9-qubit Shor-type code)
-> h2, and the `fix` callback is never touched.

Run:  PYTHONPATH=src python examples/shor_by_composition.py
"""

from wrap import Callbacks, Gate, Kernel, Pipeline
from wrap.backends import Emulator, PauliNoise, StimBackend
from wrap.frontend.toy import allocate, entangle, measure, mix, to_program, toy
from wrap.passes import BitFlipRep3, CliffordToH2, PhaseFlipRep3, ToyToClifford


@toy
def main():
    q1 = allocate()
    q2 = allocate()
    q3 = allocate()
    mix(q2)                     # q2 -> |+>
    entangle(q2, q3)            # Bell pair (q2, q3)
    measure(q3)                 # m3
    measure(q2, callback="fix", params=(q1,))   # if m2 == 1: flip q1
    measure(q1)                 # m1 == m2 after fixup, deterministically


def fix(mstack, *, q0):
    """Written once against the TOY gate set; works unchanged at every stage."""
    if mstack.bits[-1] == 1:
        return Kernel(body=[Gate("flip", (q0,))])
    return None


program = to_program(main)
callbacks = Callbacks().with_(fix=fix)

pipeline = Pipeline([ToyToClifford(), BitFlipRep3(), PhaseFlipRep3(), CliffordToH2()])

print("=" * 72)
for name, prog, cbs in pipeline.stages(program, callbacks):
    nq = len({q for q in prog.kernel.qubits()})
    ng = sum(1 for _ in prog.kernel.gates())
    print(f"{name:>16} | gateset={prog.gateset.name:<9} qubits={nq:<3} gates={ng:<4} callbacks={sorted(cbs)}")
print("=" * 72)

# Semantic preservation: identical logical distributions at every stage.
# State-vector emulation up to 9 live qubits; Stim (Pauli-frame) for the
# 27-qubit Clifford stage.  The H2 stage is checked on the 9-qubit pipeline.
emu = Emulator(seed=1)
stages = list(pipeline.stages(program, callbacks))
for name, prog, cbs in stages[:3]:
    counts = emu.sample(prog, cbs, shots=400)
    print(f"{name:>16} [statevector] -> {dict(sorted(counts.items()))}")
name, prog, cbs = stages[3]
counts = StimBackend().sample(prog, cbs, shots=400)
print(f"{name:>16} [stim]        -> {dict(sorted(counts.items()))}")
h2_prog, h2_cbs = Pipeline([ToyToClifford(), BitFlipRep3(), CliffordToH2()]).compile(program, callbacks)
counts = emu.sample(h2_prog, h2_cbs, shots=400)
print(f"{'rep3-bit->h2':>16} [statevector] -> {dict(sorted(counts.items()))}")

# The invariant: m1 == m2 (bits are bottom->top = [m3, m2, m1]).
print("\nNoise demo (logical failure = m1 != m2), Stim backend, p=0.02 per gate:")
for name, prog, cbs in stages[1:4]:  # clifford stages only
    counts = StimBackend(p=0.02).sample(prog, cbs, shots=20000)
    fail = sum(c for k, c in counts.items() if k[1] != k[2]) / sum(counts.values())
    print(f"{name:>16} | logical failure rate = {fail:.4f}")
