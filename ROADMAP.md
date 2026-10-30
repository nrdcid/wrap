# Roadmap

Reference: Paz & Grossman, *qstack: Compositional End-to-End Compilation for
Fault-Tolerant Quantum Programs*, arXiv:2605.16595 (May 2026).

## 0. Scaffold (done)
- Kernel IR, measurement stack, typed `Decoder` arity
- `Compiler` base with callback wrapping; `Pipeline`
- Passes: toy→clifford, rep3-bit, rep3-phase (Shor by composition), clifford→h2
- Backends: state-vector emulator (mid-circuit callbacks, Pauli noise), Stim (static sampler + TableauSimulator dynamic path)
- Kirin front-end dialect (`wrap.toy`) → kernel IR
- Hypothesis tests: every stage agrees with the source program

## 1. Core hardening
- [ ] Measurement-stack *type* for callbacks (declare how many bits a callback consumes) and a static check that wrapping arities line up
- [ ] Clifford dialect in Kirin; run passes as Kirin rewrites instead of a parallel tree walk
- [ ] Structured control flow in the front-end (`if m: ...` lowered to callbacks)
- [ ] OpenQASM 3 export: `def` as the wrapping boundary, `extern` for decoders (the path the paper sketches)

## 2. Codes nobody has written as passes
- [ ] Steane [[7,1,3]] with syndrome-table decoder
- [ ] Rotated surface code d=3 with PyMatching decoder as callback (`passes/surface.py`)
- [ ] DETECTOR / OBSERVABLE_INCLUDE emission from QEC passes → `detector_error_model()`
- [ ] Color code via Chromobius
- [ ] Code switching / concatenation benchmarks

## 3. Verification
- [ ] Stabilizer-tableau equivalence between stages for Clifford pipelines
- [ ] Differential testing vs Qiskit/pytket on exported QASM3
- [ ] Mechanised proof of the wrapping lemma (Lean or Coq) — one proof covers every pass

## 4. Reach
- [ ] Browser playground (Pyodide): drag passes, watch qubit count and logical error update
- [ ] Pass registry (`wrap-codes`) for third-party code + decoder plug-ins
- [ ] QIR / CUDA-Q export
- [ ] Hardware run via QASM3 (IBM dynamic circuits, Quantinuum)
