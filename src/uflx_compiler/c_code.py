import re
import sys
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field

import numpy as np

from uflx_spec_interpreter._src.nodes import (
    Absolute,
    ComponentTensor,
    Conjugate,
    Determinant,
    Division,
    Float,
    GatheredIndex,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Inverse,
    Power,
    Product,
    RealPart,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from uflx_toy.nodes import RuntimeArray

from .folding import fold


@dataclass
class _Scope:
    """Buffered statements and shared expressions at one lexical scope."""

    parent: "_Scope | None" = None
    header: str = ""
    index: str | None = None
    conditional: bool = False
    statements: list = field(default_factory=list)
    expressions: dict = field(default_factory=dict)

    def lines(self, depth=1):
        for statement in self.statements:
            if isinstance(statement, _Scope):
                yield "  " * depth + statement.header + " {"
                yield from statement.lines(depth + 1)
                yield "  " * depth + "}"
            else:
                yield "  " * depth + statement


def c_kernel(e, name, read, *, output_indices=(), index_bindings=None):
    # Schedule reductions and scalar arithmetic by their bound-index dependencies.
    # read(label, offset) supplies a C array read; offset is an integer or a C expression.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    root = scope = _Scope()
    tables = {}
    folded = fold(e, preserve_indices=(*output_indices, *(index_bindings or {})))
    serial = 0

    def fresh(prefix):
        nonlocal serial
        name = f"{prefix}{serial}"
        serial += 1
        return name

    def emit(code):
        scope.statements.append(code)

    @contextmanager
    def block(header, *, index=None, conditional=False):
        nonlocal scope
        parent = scope
        child = _Scope(parent, header, index, conditional)
        scope = child
        try:
            yield
        finally:
            scope = parent
            # Hoisted statements have already been added to the parent.
            parent.statements.append(child)

    def temp(code):
        name = fresh("t")
        emit(f"const double {name} = {code};")
        return name

    def entry(n, bound, comp):
        nonlocal scope
        positions = tuple(bound[i] for i, _ in n.free_indices)
        key = (n, positions, comp)
        # Bound positions may be C expressions, including gathered table indices.
        names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", " ".join(map(str, positions + comp))))
        target = scope
        while target.parent is not None and not target.conditional and target.index not in names:
            target = target.parent
        ancestor = target
        while ancestor is not None:
            if key in ancestor.expressions:
                return ancestor.expressions[key]
            ancestor = ancestor.parent
        previous = scope
        scope = target
        try:
            result = scalar(n, bound, comp)
            target.expressions[key] = result
            return result
        finally:
            scope = previous

    def table(key, values, indices, dtype="double"):
        if all(isinstance(i, int) for i in indices):
            return float(values[indices]) if dtype == "double" else int(values[indices])
        if key not in tables:
            name = fresh("table")
            shape = "".join(f"[{n}]" for n in values.shape)
            tables[key] = (name, f"  static const {dtype} {name}{shape} = {initializer(values)};")
        name = tables[key][0]
        return name + "".join(f"[{i}]" for i in indices)

    def initializer(values):
        if values.ndim == 0:
            return repr(values.item())
        return "{" + ", ".join(initializer(v) for v in values) + "}"

    def offset(comp, shape):
        result = 0
        for i, n in zip(comp, shape, strict=True):
            if result == 0:
                result = i
            elif isinstance(result, int) and isinstance(i, int):
                result = result * n + i
            else:
                result = f"({result} * {n} + {i})"
        return result

    def scalar(n, bound, comp):
        if n in folded:
            indices = tuple(bound[i] for i, _ in n.free_indices) + comp
            return table(n, folded[n], indices)
        match n:
            case Integer(v) | Float(v):
                return float(v)
            case Zero():
                return 0.0
            case RuntimeArray(label, shape):
                return read(label, offset(comp, shape))
            case Sum(a, b):
                return add(entry(a, bound, comp), entry(b, bound, comp))
            case Product(a, b) if a.value_shape:
                return multiply(entry(a, bound, comp), entry(b, bound, ()))
            case Product(a, b):
                return multiply(entry(a, bound, ()), entry(b, bound, comp))
            case Division(a, b):
                return divide(entry(a, bound, comp), entry(b, bound, ()))
            case Power(a, Integer(k)) if k > 0:
                x = entry(a, bound, comp)
                out = x
                for _ in range(k - 1):
                    out = multiply(out, x)
                return out
            case Absolute(a):
                return call("fabs", np.abs, entry(a, bound, comp))
            case SquareRoot(a):
                return call("sqrt", np.sqrt, entry(a, bound, comp))
            case Conjugate(a) | RealPart(a):
                return entry(a, bound, comp)
            case Tensor(entries):
                if isinstance(comp[0], int):
                    return entry(entries[comp[0]], bound, comp[1:])
                result = fresh("t")
                emit(f"double {result};")
                with block(f"switch ({comp[0]})"):
                    for k, x in enumerate(entries):
                        with block(f"case {k}:", conditional=True):
                            emit(f"{result} = {render(entry(x, bound, comp[1:]))};")
                            emit("break;")
                return result
            case Indexed(x, indices):
                return entry(x, bound, tuple(position(k, bound) for k in indices) + comp)
            case ComponentTensor(x, indices):
                inner = bound | dict(zip(indices, comp, strict=False))
                return entry(x, inner, comp[len(indices) :])
            case IndexSum(x, indices):
                extent = dict(x.free_indices)
                if any(extent[i] == 0 for i in indices):
                    return 0.0
                total = fresh("sum")
                emit(f"double {total} = 0.0;")
                inner = dict(bound)
                with ExitStack() as stack:
                    for i in indices:
                        name = fresh("i")
                        stack.enter_context(
                            block(
                                f"for (int {name} = 0; {name} < {extent[i]}; ++{name})", index=name
                            )
                        )
                        inner[i] = name
                    emit(f"{total} += {render(entry(x, inner, comp))};")
                return total
            case Determinant(a):
                size = a.value_shape[0]
                return determinant(
                    [[entry(a, bound, (i, j)) for j in range(size)] for i in range(size)]
                )
            case Inverse(a) if a.value_shape in ((1, 1), (2, 2), (3, 3)):
                return inverse(a, bound, comp)
        raise TypeError(f"the C backend cannot write {type(n).__name__}")

    def position(k, bound):
        match k:
            case Index():
                return bound[k]
            case GatheredIndex(t, indices):
                positions = tuple(position(i, bound) for i in indices)
                return table(t, np.asarray(t.values), positions, "int")
        return int(k)

    def inverse(a, bound, comp):
        n = a.value_shape[0]
        A = [[entry(a, bound, (r, c)) for c in range(n)] for r in range(n)]
        det = determinant(A)
        if n == 1:
            return divide(1.0, det)

        def cofactor(i, j):
            minor = [[A[r][c] for c in range(n) if c != i] for r in range(n) if r != j]
            value = determinant(minor)
            return value if (i + j) % 2 == 0 else negate(value)

        if all(isinstance(i, int) for i in comp):
            return divide(cofactor(*comp), det)
        rows = ["{" + ", ".join(render(cofactor(i, j)) for j in range(n)) + "}" for i in range(n)]
        name = fresh("cofactors")
        emit(f"const double {name}[{n}][{n}] = {{" + ", ".join(rows) + "};")
        return divide(f"{name}[{comp[0]}][{comp[1]}]", det)

    def determinant(A):
        if len(A) == 1:
            return A[0][0]
        if len(A) == 2:
            return add(multiply(A[0][0], A[1][1]), negate(multiply(A[0][1], A[1][0])))
        total = 0.0
        for c in range(3):
            minor = [[A[r][k] for k in range(3) if k != c] for r in (1, 2)]
            term = multiply(A[0][c], determinant(minor))
            total = add(total, term if c % 2 == 0 else negate(term))
        return total

    def add(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return a + b
        if a == 0.0:
            return b
        if b == 0.0:
            return a
        return temp(f"{render(a)} + {render(b)}")

    def multiply(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return a * b
        if a == 0.0 or b == 0.0:
            return 0.0
        if a == 1.0:
            return b
        if b == 1.0:
            return a
        return temp(f"{render(a)} * {render(b)}")

    def divide(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return a / b
        if a == 0.0:
            return 0.0
        return temp(f"{render(a)} / {render(b)}")

    def negate(a):
        return -a if isinstance(a, float) else temp(f"-{a}")

    def call(name, f, a):
        return float(f(a)) if isinstance(a, float) else temp(f"{name}({a})")

    def accumulate(n, bound):
        # Integral sums surround argument loops. Other reductions remain scalar
        # computations, which entry() schedules at their earliest legal scope.
        if isinstance(n, Sum):
            accumulate(n.first, bound)
            accumulate(n.second, bound)
            return
        if isinstance(n, IndexSum):
            extent = dict(n.expression.free_indices)
            with ExitStack() as stack:
                inner = dict(bound)
                for i in n.indices:
                    if extent[i] == 0:
                        return
                    loop = fresh("q")
                    stack.enter_context(
                        block(f"for (int {loop} = 0; {loop} < {extent[i]}; ++{loop})", index=loop)
                    )
                    inner[i] = loop
                accumulate(n.expression, inner)
            return
        with ExitStack() as stack:
            inner = dict(bound)
            for i in output_indices:
                loop = fresh("a")
                stack.enter_context(
                    block(
                        f"for (int {loop} = 0; {loop} < {dict(e.free_indices)[i]}; ++{loop})",
                        index=loop,
                    )
                )
                inner[i] = loop
            result = entry(n, inner, ())
            output = offset(
                tuple(inner[i] for i in output_indices),
                tuple(dict(e.free_indices)[i] for i in output_indices),
            )
            emit(f"A[{output}] += {render(result)};")

    bound = {i: 0 for i, _ in e.free_indices} | (index_bindings or {})
    if all(dict(e.free_indices)[i] > 0 for i in output_indices):
        accumulate(e, bound)
    body = "\n".join([declaration for _, declaration in tables.values()] + list(root.lines()))
    return (
        "#include <stdint.h>\n"
        f"void {name}(double* restrict A, const double* restrict w, const double* restrict c,\n"
        "    const double* restrict coordinate_dofs, const int* restrict entity_local_index,\n"
        "    const uint8_t* restrict quadrature_permutation, void* custom_data)\n"
        f"{{\n{body}\n}}\n"
    )


def render(x):
    return repr(x) if isinstance(x, float) else x
