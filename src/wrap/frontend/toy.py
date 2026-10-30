"""Kirin front-end: write toy-ISA programs as decorated Python functions.

    from wrap.frontend.toy import toy, allocate, flip, mix, entangle, measure

    @toy
    def main():
        q1 = allocate()
        q2 = allocate()
        mix(q2)
        entangle(q2, q1)
        measure(q2)
        measure(q1, callback="fix", params=(q1,))

    program = to_program(main)

Kirin lowers the function to SSA; `to_program` walks the straight-line body
and rebuilds the kernel tree, enforcing the LIFO allocate/measure discipline.

TODO(roadmap): structured control flow (scf) -> callbacks; type-checked
callback parameters; a dialect for CLIFFORD so passes can run *inside* Kirin
as rewrite rules.
"""

from __future__ import annotations

import ast

from kirin import ir, lowering, types
from kirin.decl import info, statement
from kirin.dialects import func
from kirin.dialects.py import constant
from kirin.prelude import basic_no_opt

from ..ir import TOY, CallbackRef, Gate, Kernel, Program, Qubit

dialect = ir.Dialect("wrap.toy")


class QubitHandle:
    """Front-end stand-in for a qubit SSA value."""


QubitT = types.PyClass(QubitHandle)


@statement(dialect=dialect)
class Allocate(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    result: ir.ResultValue = info.result(QubitT)


@statement(dialect=dialect)
class Flip(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    qubit: ir.SSAValue = info.argument(QubitT)


@statement(dialect=dialect)
class Mix(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    qubit: ir.SSAValue = info.argument(QubitT)


@statement(dialect=dialect)
class Entangle(ir.Statement):
    traits = frozenset({lowering.FromPythonCall()})
    control: ir.SSAValue = info.argument(QubitT)
    target: ir.SSAValue = info.argument(QubitT)


class _MeasureLowering(lowering.FromPythonCall):
    """Lower `measure(q)`, `measure(q, callback="fix")`, or
    `measure(q, callback="fix", params=(q1, q2))`; `params` defaults to ()."""

    def lower(self, stmt, state, node: ast.Call):
        if not any(kw.arg == "params" for kw in node.keywords):
            node = ast.Call(
                node.func,
                list(node.args),
                list(node.keywords) + [ast.keyword("params", ast.Tuple([], ast.Load()))],
            )
        return super().lower(stmt, state, node)


@statement(dialect=dialect)
class Measure(ir.Statement):
    traits = frozenset({_MeasureLowering()})
    qubit: ir.SSAValue = info.argument(QubitT)
    callback: str = info.attribute(default="done")
    params: tuple[ir.SSAValue, ...] = info.argument(QubitT)
    """Qubit parameters forwarded to the callback as `q0=, q1=, ...`."""


@lowering.wraps(Allocate)
def allocate() -> QubitHandle: ...


@lowering.wraps(Flip)
def flip(qubit: QubitHandle) -> None: ...


@lowering.wraps(Mix)
def mix(qubit: QubitHandle) -> None: ...


@lowering.wraps(Entangle)
def entangle(control: QubitHandle, target: QubitHandle) -> None: ...


@lowering.wraps(Measure)
def measure(qubit: QubitHandle, callback: str = "done", params: tuple[QubitHandle, ...] = ()) -> None: ...


toy = basic_no_opt.add(dialect)
"""Decorator: Python function -> Kirin Method in the toy dialect."""


_GATES = {Flip: "flip", Mix: "mix", Entangle: "entangle"}


def to_program(method: ir.Method) -> Program:
    """Rebuild the kernel tree from a straight-line Kirin method."""
    names: dict[ir.SSAValue, Qubit] = {}
    root = Kernel()
    stack: list[Kernel] = [root]
    n = 0

    for stmt in method.callable_region.blocks[0].stmts:
        if isinstance(stmt, Allocate):
            n += 1
            q = Qubit(f"q{n}")
            names[stmt.result] = q
            k = Kernel(target=q)
            stack[-1].body.append(k)
            stack.append(k)
        elif type(stmt) in _GATES:
            stack[-1].body.append(Gate(_GATES[type(stmt)], tuple(names[a] for a in stmt.args)))
        elif isinstance(stmt, Measure):
            q = names[stmt.qubit]
            if stack[-1].target != q:
                raise ValueError(
                    f"measure {q} violates LIFO: innermost open allocation is {stack[-1].target}"
                )
            params = {f"q{i}": names[p] for i, p in enumerate(stmt.params)}
            stack[-1].callback = CallbackRef(stmt.callback, params)
            stack.pop()
        elif isinstance(stmt, (func.Return, func.ConstantNone, constant.Constant)):
            continue
        else:
            raise NotImplementedError(f"unsupported statement in toy front-end: {stmt.name}")

    if len(stack) != 1:
        raise ValueError(f"{len(stack) - 1} qubit(s) never measured: {[k.target for k in stack[1:]]}")
    # A single top-level kernel is the program; multiple become a compute block.
    kernel = root.body[0] if len(root.body) == 1 and isinstance(root.body[0], Kernel) else root
    return Program(TOY, kernel)
