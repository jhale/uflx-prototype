import re
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field
from functools import cache

import numpy as np

from uflx_spec_interpreter._src.algorithms.substitution import substitute
from uflx_spec_interpreter._src.nodes import (
    Absolute,
    BroadcastTo,
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
    Logarithm,
    Power,
    Product,
    RealPart,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from uflx_spec_interpreter._src.nodes.structure import expression_operands
from uflx_toy.nodes import RuntimeArray

from .contractions import _factor_contraction
from .folding import fold
from .simplification import simplify


@dataclass
class _ProductStatement:
    """A scalar product retained until its uses are known."""

    name: str
    expression: str

    def __str__(self):
        return f"const double {self.name} = {self.expression};"


@dataclass
class _Accumulation:
    """An in-place scalar addition."""

    target: str
    value: str

    def __str__(self):
        return f"{self.target} += {self.value};"


@dataclass
class _Scope:
    """Buffered statements and shared expressions at one lexical scope."""

    parent: "_Scope | None" = None
    header: str = ""
    index: str | None = None
    conditional: bool = False
    statements: list = field(default_factory=list)
    expressions: dict = field(default_factory=dict)

    @property
    def depth(self):
        return 0 if self.parent is None else self.parent.depth + 1

    def lines(self, depth=1):
        for statement in self.statements:
            if isinstance(statement, _Scope):
                yield "  " * depth + statement.header + " {"
                yield from statement.lines(depth + 1)
                yield "  " * depth + "}"
            else:
                yield "  " * depth + str(statement)

    def inline_products(self):
        """Expose multiply-adds without duplicating shared or hoisted products."""
        uses = Counter(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", "\n".join(self.lines())))

        def visit(scope):
            statements = []
            for statement in scope.statements:
                if isinstance(statement, _Scope):
                    visit(statement)
                if isinstance(statement, _Accumulation) and statements:
                    previous = statements[-1]
                    if (
                        isinstance(previous, _ProductStatement)
                        and statement.value == previous.name
                        and uses[previous.name] == 2  # Declaration and sole use.
                    ):
                        statements.pop()
                        statement = _Accumulation(statement.target, previous.expression)
                statements.append(statement)
            scope.statements = statements

        visit(self)


def c_kernel(e, name, read, *, output_indices=(), index_bindings=None, unroll_indices=()):
    # Schedule reductions and scalar arithmetic by their bound-index dependencies.
    # read(label, offset) supplies a C array read; offset is an integer or a C expression.
    sys.setrecursionlimit(max(sys.getrecursionlimit(), 100000))
    root = scope = _Scope()
    tables = {}
    if not set(unroll_indices) <= set(output_indices):
        raise ValueError("unrolled indices must be output indices")
    extents = dict(e.free_indices)
    # Specialise component choices before scheduling, exposing exact structural zeros.
    expressions = []
    folded = {}
    for components in np.ndindex(tuple(extents[i] for i in unroll_indices)):
        selected = dict(zip(unroll_indices, components, strict=True))
        expression = simplify(substitute(e, selected)) if selected else e
        expressions.append((selected, expression))
        folded.update(fold(expression, preserve_indices=(*output_indices, *(index_bindings or {}))))
    loop_indices = tuple(i for i in output_indices if i not in unroll_indices)
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

    def placement(n, bound, comp=()):
        # Bound positions may be C expressions, including gathered table indices.
        positions = tuple(bound[i] for i, _ in n.free_indices)
        names = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", " ".join(map(str, positions + comp))))
        target = scope
        while target.parent is not None and not target.conditional and target.index not in names:
            target = target.parent
        return target

    def entry(n, bound, comp):
        nonlocal scope
        positions = tuple(bound[i] for i, _ in n.free_indices)
        key = (n, positions, comp)
        target = placement(n, bound, comp)
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

    def ordered_product(n, bound):
        factors = []

        def collect(x):
            if isinstance(x, Product) and not x.value_shape:
                collect(x.first)
                collect(x.second)
            else:
                factors.append(x)

        collect(n)
        # Multiply factors available in outer scopes first, exposing reusable
        # weights and geometry before argument-dependent basis products.
        factors.sort(key=lambda factor: placement(factor, bound).depth)
        result = factors[0]
        for factor in factors[1:]:
            result = Product(result, factor)
        return result

    @cache
    def selected_indices(n):
        if n in folded:
            return frozenset()
        indices = set()
        if isinstance(n, Indexed) and isinstance(n.expression, Tensor):
            indices.update(i for i in n.indices if isinstance(i, Index))
        for child in expression_operands(n):
            indices.update(selected_indices(child))
        return frozenset(indices)

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
                if not n.value_shape:
                    ordered = ordered_product(n, bound)
                    if ordered != n:
                        return entry(ordered, bound, comp)
                return multiply(entry(a, bound, ()), entry(b, bound, comp))
            case Division(a, b):
                return divide(entry(a, bound, comp), entry(b, bound, ()))
            case Power(a, Integer(k)) if k > 0:
                x = entry(a, bound, comp)
                out = x
                for _ in range(k - 1):
                    out = multiply(out, x)
                return out
            case Power(a, b):
                return call("pow", np.float_power, entry(a, bound, comp), entry(b, bound, ()))
            case Logarithm(a):
                return call("log", np.log, entry(a, bound, comp))
            case Absolute(a):
                return call("fabs", np.abs, entry(a, bound, comp))
            case SquareRoot(a):
                return call("sqrt", np.sqrt, entry(a, bound, comp))
            case Conjugate(a) | RealPart(a):
                return entry(a, bound, comp)
            case BroadcastTo(x, _):
                leading = len(comp) - len(x.value_shape)
                selected = tuple(
                    0 if n == 1 else comp[leading + k] for k, n in enumerate(x.value_shape)
                )
                return entry(x, bound, selected)
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

                def reduce(indices, inner):
                    if not indices:
                        emit(_Accumulation(total, render(entry(x, inner, comp))))
                        return
                    i, *rest = indices
                    # Resolve small spatial component selections before scheduling;
                    # quadrature and DOF reductions retain their loops.
                    if extent[i] <= 3 and i in selected_indices(x):
                        for k in range(extent[i]):
                            reduce(rest, inner | {i: k})
                    else:
                        name = fresh("i")
                        with block(
                            f"for (int {name} = 0; {name} < {extent[i]}; ++{name})", index=name
                        ):
                            reduce(rest, inner | {i: name})

                reduce(indices, bound)
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
        name = fresh("t")
        emit(_ProductStatement(name, f"{render(a)} * {render(b)}"))
        return name

    def divide(a, b):
        if isinstance(a, float) and isinstance(b, float):
            return a / b
        if a == 0.0:
            return 0.0
        return temp(f"{render(a)} / {render(b)}")

    def negate(a):
        return -a if isinstance(a, float) else temp(f"-{a}")

    def call(name, f, *args):
        if all(isinstance(a, float) for a in args):
            return float(f(*args))
        return temp(f"{name}({', '.join(map(render, args))})")

    @cache
    def work(n):
        if n in folded or isinstance(n, RuntimeArray):
            return 0
        arithmetic = isinstance(n, (Sum, Product, Division, Power, IndexSum, Inverse, Determinant))
        return int(arithmetic) + sum(work(child) for child in expression_operands(n))

    def prepare(n, bound):
        # Materialise maximal computations reused across an output index.
        # Precomputing both sides also separates basis evaluation from contraction.
        # Arrays span only the indices the computation depends on.
        if len(loop_indices) < 2:
            return
        seen = set()

        def visit(n):
            if n in seen or n in folded:
                return
            seen.add(n)
            free = dict(n.free_indices)
            if free.keys() <= bound.keys():
                key = (n, tuple(bound[i] for i, _ in n.free_indices), ())
                ancestor = scope
                while ancestor is not None:
                    if key in ancestor.expressions:
                        return
                    ancestor = ancestor.parent
            axes = tuple(i for i in loop_indices if i in free)
            if (
                axes
                and not n.value_shape
                and free.keys() <= bound.keys()
                and work(n) > 1
                and any(dict(e.free_indices)[i] > 1 for i in loop_indices if i not in free)
            ):
                name = fresh("cache")
                emit(f"double {name}" + "".join(f"[{free[i]}]" for i in axes) + ";")
                with ExitStack() as stack:
                    inner = dict(bound)
                    for i in axes:
                        loop = fresh("p")
                        stack.enter_context(
                            block(f"for (int {loop} = 0; {loop} < {free[i]}; ++{loop})", index=loop)
                        )
                        inner[i] = loop
                    value = entry(n, inner, ())
                    access = name + "".join(f"[{inner[i]}]" for i in axes)
                    emit(f"{access} = {render(value)};")
                key = (n, tuple(bound[i] for i, _ in n.free_indices), ())
                scope.expressions[key] = name + "".join(f"[{bound[i]}]" for i in axes)
                return
            if isinstance(n, (Tensor, IndexSum, ComponentTensor)):
                return  # Do not cross conditional branches or local index bindings.
            for child in expression_operands(n):
                visit(child)

        visit(n)

    def accumulate(n, bound):
        if isinstance(n, Zero):
            return
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
        emit_blocks([({}, n)], bound)

    def emit_blocks(blocks, bound):
        if len(loop_indices) == 2:
            outer_indices = tuple(i for i in bound if i not in output_indices)
            # Coupled component blocks share physical derivatives; independently
            # forming a metric for every block would duplicate that preparation.
            combine_geometry = len(blocks) == 1
            blocks = [
                (
                    selected,
                    _factor_contraction(
                        n, loop_indices, outer_indices, combine_geometry=combine_geometry
                    ),
                )
                for selected, n in blocks
            ]
            for _, n in blocks:
                folded.update(fold(n, preserve_indices=(*output_indices, *(index_bindings or {}))))
        # Prepare all components together, then update adjacent entries in one
        # basis-pair traversal. This exposes interleaved stores to the C compiler.
        with ExitStack() as stack:
            inner = bound | loop_names
            for selected, n in blocks:
                prepare(n, inner | selected)
            for i in loop_indices:
                loop = inner[i]
                stack.enter_context(
                    block(
                        f"for (int {loop} = 0; {loop} < {dict(e.free_indices)[i]}; ++{loop})",
                        index=loop,
                    )
                )
            for selected, n in blocks:
                positions = inner | selected
                result = entry(n, positions, ())
                output = offset(
                    tuple(positions[i] for i in output_indices),
                    tuple(dict(e.free_indices)[i] for i in output_indices),
                )
                emit(_Accumulation(f"A[{output}]", render(result)))

    def accumulate_blocks(blocks, bound):
        # Share a common quadrature traversal across component blocks, so geometry
        # and basis computations can be reused at each point.
        blocks = [(selected, n) for selected, n in blocks if not isinstance(n, Zero)]
        if not blocks:
            return
        first = blocks[0][1]
        if isinstance(first, IndexSum):
            extents = dict(first.expression.free_indices)
            axes = tuple((i, extents[i]) for i in first.indices)
            if all(
                isinstance(n, IndexSum)
                and tuple((i, dict(n.expression.free_indices)[i]) for i in n.indices) == axes
                for _, n in blocks
            ):
                with ExitStack() as stack:
                    inner = dict(bound)
                    for i, size in axes:
                        if size == 0:
                            return
                        loop = fresh("q")
                        stack.enter_context(
                            block(f"for (int {loop} = 0; {loop} < {size}; ++{loop})", index=loop)
                        )
                        inner[i] = loop
                    accumulate_blocks([(selected, n.expression) for selected, n in blocks], inner)
                return
        if all(not isinstance(n, (Sum, IndexSum)) for _, n in blocks):
            emit_blocks(blocks, bound)
        else:
            for selected, n in blocks:
                accumulate(n, bound | selected)

    # Reusing loop names also lets prepared basis arrays match across component blocks.
    loop_names = {i: fresh("a") for i in loop_indices}
    bound = {i: 0 for i, _ in e.free_indices} | (index_bindings or {})
    if all(dict(e.free_indices)[i] > 0 for i in output_indices):
        accumulate_blocks(expressions, bound)
    root.inline_products()
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
