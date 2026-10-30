"""Stim backend for CLIFFORD programs.

Two execution paths:

* **static** (`sample`): emit the circuit assuming callbacks return no kernel,
  sample records with Stim's Pauli-frame sampler, then replay the callback
  chain offline on each record.  Wrapped callbacks are just decoders, so this
  reproduces the full pipeline's logical outcomes at Stim speed.
* **dynamic** (`run`): drive `stim.TableauSimulator` interactively, invoking
  callbacks mid-circuit exactly like the state-vector emulator.  `sample`
  falls back to this when a callback returns a kernel.

TODO(roadmap): emit DETECTOR / OBSERVABLE_INCLUDE annotations from QEC
passes so `stim.Circuit.detector_error_model()` and PyMatching work directly.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import stim

from ..callbacks import Callbacks
from ..ir import CLIFFORD, Gate, Kernel, Program, Qubit
from ..mstack import MeasurementStack


class DynamicProgramError(RuntimeError):
    pass


@dataclass
class StimBackend:
    #: per-gate depolarising noise strength (0 = noiseless)
    p: float = 0.0

    def circuit(self, program: Program) -> tuple[stim.Circuit, list[tuple[Qubit, str, dict]]]:
        """Build the static circuit.  Returns (circuit, measurement log) where
        the log lists, in measurement order, (qubit, callback name, params)."""
        if program.gateset != CLIFFORD:
            raise ValueError(f"Stim backend needs CLIFFORD programs, got {program.gateset.name!r}")
        c = stim.Circuit()
        idx: dict[Qubit, int] = {}
        log: list[tuple[Qubit, str, dict]] = []
        counter = [0]

        def alloc(q: Qubit) -> int:
            i = counter[0]
            counter[0] += 1
            idx[q] = i
            c.append("R", [i])
            return i

        def walk(k: Kernel) -> None:
            if k.target is not None:
                alloc(k.target)
            for ins in k.body:
                if isinstance(ins, Gate):
                    targets = [idx[q] for q in ins.qubits]
                    c.append(ins.name, targets)
                    if self.p > 0:
                        c.append("DEPOLARIZE1" if len(targets) == 1 else "DEPOLARIZE2", targets, self.p)
                else:
                    walk(ins)
            if k.target is not None:
                c.append("M", [idx[k.target]])
                log.append((k.target, k.callback.name, dict(k.callback.params)))

        walk(program.kernel)
        return c, log

    # ---- dynamic path -----------------------------------------------------------
    def run(self, program: Program, callbacks: Callbacks | None = None, seed: int | None = None) -> list[int]:
        """Single shot with mid-circuit callbacks via TableauSimulator."""
        if program.gateset != CLIFFORD:
            raise ValueError(f"Stim backend needs CLIFFORD programs, got {program.gateset.name!r}")
        cbs = callbacks or Callbacks()
        sim = stim.TableauSimulator(seed=seed)
        idx: dict[Qubit, int] = {}
        free: list[int] = []
        counter = [0]
        ms = MeasurementStack()

        def alloc(q: Qubit) -> None:
            i = free.pop() if free else counter[0]
            if not free and i == counter[0]:
                counter[0] += 1
            idx[q] = i
            sim.reset(i)

        def kernel(k: Kernel) -> None:
            if k.target is not None:
                alloc(k.target)
            for ins in k.body:
                if isinstance(ins, Gate):
                    t = [idx[q] for q in ins.qubits]
                    getattr(sim, ins.name.lower())(*t)
                    if self.p > 0:
                        (sim.depolarize1 if len(t) == 1 else sim.depolarize2)(*t, p=self.p)
                else:
                    kernel(ins)
            if k.target is None:
                return
            i = idx.pop(k.target)
            ms.push(int(sim.measure(i)))
            free.append(i)
            new = cbs[k.callback.name](ms, **k.callback.params)
            if new is not None:
                program.gateset.check(new)
                kernel(new)

        kernel(program.kernel)
        return list(ms.bits)

    def sample(self, program: Program, callbacks: Callbacks | None = None, shots: int = 1000, seed: int | None = None) -> dict[tuple[int, ...], int]:
        """Static fast path; falls back to per-shot TableauSimulator if dynamic."""
        try:
            return self._sample_static(program, callbacks, shots)
        except DynamicProgramError:
            counts: dict[tuple[int, ...], int] = {}
            rng = np.random.default_rng(seed)
            for _ in range(shots):
                k = tuple(self.run(program, callbacks, seed=int(rng.integers(2**31))))
                counts[k] = counts.get(k, 0) + 1
            return counts

    def _sample_static(self, program: Program, callbacks: Callbacks | None, shots: int) -> dict[tuple[int, ...], int]:
        cbs = callbacks or Callbacks()
        circuit, log = self.circuit(program)
        records = circuit.compile_sampler().sample(shots)
        counts: dict[tuple[int, ...], int] = {}
        for rec in records:
            ms = MeasurementStack()
            for bit, (_q, cb, params) in zip(rec, log):
                ms.push(int(bit))
                new = cbs[cb](ms, **params)
                if new is not None:
                    raise DynamicProgramError(
                        f"callback {cb!r} returned a kernel; program is dynamic. Use backends.Emulator."
                    )
            key = tuple(ms.bits)
            counts[key] = counts.get(key, 0) + 1
        return counts
