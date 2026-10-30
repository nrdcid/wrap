# wrap

**QEC is just a compiler pass.**

`wrap` is an open fault-tolerant quantum compiler stack built on *callback
wrapping*, the mechanism introduced by Paz & Grossman in
[qstack: Compositional End-to-End Compilation for Fault-Tolerant Quantum Programs](https://arxiv.org/abs/2605.16595)
(arXiv:2605.16595, May 2026).

The IR is purely quantum. All classical logic (decoders, fixups,
repeat-until-success) lives in opaque **callbacks**. When a pass transforms a
program it also wraps every callback so that it (1) sees correctly decoded
measurement bits and (2) has any kernel it returns compiled through the same
pass. Consequently:

- **ISA translation and QEC are the same kind of object** — a `Compiler` with gate handlers and a `decode` step.
- **Passes compose by function composition.** `BitFlipRep3` then `PhaseFlipRep3` is a 9-qubit Shor-type code. Nobody wrote a Shor pass.
- **Callbacks are written once.** A fixup written against the toy ISA runs unchanged after four compilation stages.

```python
from wrap import Callbacks, Gate, Kernel, Pipeline
from wrap.frontend.toy import toy, allocate, mix, entangle, measure, to_program
from wrap.passes import ToyToClifford, BitFlipRep3, PhaseFlipRep3, CliffordToH2
from wrap.backends import StimBackend

@toy
def main():
    q1 = allocate()
    q2 = allocate()
    q3 = allocate()
    mix(q2); entangle(q2, q3)
    measure(q3)
    measure(q2, callback="fix", params=(q1,))   # if m2 == 1: flip q1
    measure(q1)

def fix(mstack, *, q0):                          # written against the TOY ISA, never changes
    return Kernel([Gate("flip", (q0,))]) if mstack.bits[-1] else None

program   = to_program(main)
callbacks = Callbacks().with_(fix=fix)

pipeline  = Pipeline([ToyToClifford(), BitFlipRep3(), PhaseFlipRep3(), CliffordToH2()])
compiled, compiled_callbacks = pipeline.compile(program, callbacks)   # 3 logical -> 27 physical qubits
```

Run the full demo (semantic preservation at every stage plus a noise sweep):

```
pip install -e ".[dev]"
python examples/shor_by_composition.py
pytest -q
```

## What is here

| Module | Contents |
|---|---|
| `wrap.ir` | `Kernel` / `Gate` / `CallbackRef` / `GateSet` / `Program`; `TOY`, `CLIFFORD`, `H2` gate sets |
| `wrap.mstack` | LIFO `MeasurementStack`, typed `Decoder(in_bits, out_bits)` |
| `wrap.compiler` | `Compiler` base (handlers, `encode`, `prepare`, `pre_measure`, `decoder`, callback wrapping), `Pipeline` |
| `wrap.passes` | `ToyToClifford`, `BitFlipRep3`, `PhaseFlipRep3`, `CliffordToH2`; `SurfaceCode` design stub |
| `wrap.backends` | `Emulator` (state vector, mid-circuit callbacks, Pauli noise), `StimBackend` (static Pauli-frame sampler + dynamic `TableauSimulator`) |
| `wrap.frontend.toy` | [Kirin](https://github.com/QuEraComputing/kirin) dialect: write programs as Python functions, lower to kernel IR |
| `tests/` | Unit tests, H2 unitary checks, Hypothesis tests that every stage agrees with the source program |

## Design notes beyond the paper

- **Typed decoder arity.** `Decoder(in_bits, out_bits)` is checked against `n_physical` at pass construction, so wrapping-order mistakes fail early instead of corrupting the stack.
- **Pass-unique no-op callbacks.** Inner physical kernels use `__noop@<pass>`; later passes wrap these too, which is exactly what makes concatenation work.
- **Two Stim paths.** Static programs sample at Pauli-frame speed and replay callbacks offline; dynamic programs fall back to an interactive tableau run.
- **Non-FT escape hatch.** Repetition codes have no transversal `H`; `H`/`S` are implemented by unencode–apply–re-encode and labelled as such.

See [ROADMAP.md](ROADMAP.md) for what comes next (surface code with PyMatching as a callback, Steane, QASM3 export, Stim detectors, a browser playground).

## License

Apache-2.0.
