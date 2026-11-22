import re
import sys
from collections import Counter
from contextlib import ExitStack, contextmanager
from dataclasses import dataclass, field, replace
from functools import cache
from math import prod

import numpy as np

from uflx import replace as substitute
from uflx.algorithms import expression_operands
from uflx.nodes import (
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
from uflx_basix.nodes import RuntimeArray

from .contractions import _factor_contraction, _preintegrate, _sparse_contractions, _transpose_reuse
from .folding import fold
from .simplification import simplify


@dataclass
class _ScalarStatement:
    """A named scalar computation."""

    name: str
    expression: str

    def __str__(self):
        return f"const double {self.name} = {self.expression};"


@dataclass
class _ProductStatement(_ScalarStatement):
    """A scalar product retained until its uses are known."""


@dataclass(frozen=True)
class _BasisCache:
    """Storage for a computation depending on output indices."""

    declaration: str

    def __str__(self):
        return self.declaration


@dataclass
class _Accumulation:
    """An in-place scalar addition."""

    target: str
    value: str

    def __str__(self):
        return f"{self.target} += {self.value};"


@dataclass
class _Assignment(_Accumulation):
    """A completed scalar value written to a private output array."""

    def __str__(self):
        return f"{self.target} = {self.value};"


@dataclass(frozen=True)
class _Initialisation:
    """A scalar reduction accumulator."""

    target: str

    def __str__(self):
        return f"double {self.target} = 0.0;"


@dataclass
class _Scope:
    """Buffered statements and shared expressions at one lexical scope."""

    parent: "_Scope | None" = None
    header: str = ""
    index: str | None = None
    conditional: bool = False
    statements: list = field(default_factory=list)
    expressions: dict = field(default_factory=dict)
    quadrature_extent: int | None = None
    basis: bool = False
    output_extent: int | None = None

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
                        statement = replace(statement, value=previous.expression)
                statements.append(statement)
            scope.statements = statements

        visit(self)


def _offset(comp, shape):
    """Row-major offset, shared by table access and dense-loop flattening."""
    result = 0
    for i, n in zip(comp, shape, strict=True):
        if result == 0:
            result = i
        elif isinstance(result, int) and isinstance(i, int):
            result = result * n + i
        else:
            result = f"({result} * {n} + {i})"
    return result


def _flatten_output(scope):
    """Collapse perfect output nests accessed only through their contiguous offset.

    Any separate use of an axis (including a row cache or a transposed access)
    leaves the nest intact. No division or remainder is introduced to recover axes.
    This runs after scheduling, when only rendered statements remain relevant.
    """
    nest = [scope]
    while (
        nest[-1].output_extent is not None
        and len(nest[-1].statements) == 1
        and isinstance(child := nest[-1].statements[0], _Scope)
        and child.output_extent is not None
    ):
        nest.append(child)
    if len(nest) > 1 and all(
        isinstance(s, (_ScalarStatement, _Accumulation)) for s in nest[-1].statements
    ):
        indices = tuple(s.index for s in nest)
        extents = tuple(s.output_extent for s in nest)
        index = "flat_" + indices[0]
        offset = _offset(indices, extents)
        statements = [str(s).replace(offset, index) for s in nest[-1].statements]
        uses = set(re.findall(r"[A-Za-z_][A-Za-z_0-9]*", " ".join(statements)))
        if not uses.intersection(indices):
            extent = prod(extents)
            return replace(
                scope,
                header=f"for (int {index} = 0; {index} < {extent}; ++{index})",
                index=index,
                output_extent=extent,
                statements=statements,
            )
    return replace(
        scope,
        statements=[_flatten_output(s) if isinstance(s, _Scope) else s for s in scope.statements],
    )


def _split_quadrature(scope):
    """Separate pointwise preparation from basis loops, with bounded temporary storage.

    Scheduling has already placed computations independent of output indices in
    the quadrature scope. Only basis caches and basis loops depend on those indices;
    scalar reductions in the preparation retain their original order.
    """
    statements = [
        item
        for statement in scope.statements
        for item in (_split_quadrature(statement) if isinstance(statement, _Scope) else [statement])
    ]
    scope = replace(scope, statements=statements)
    extent = scope.quadrature_extent
    if (
        extent is None
        or extent < 2
        or not any(isinstance(s, _Scope) and s.basis for s in statements)
    ):
        return [scope]
    preparation, contraction = [], []
    for statement in statements:
        if isinstance(statement, _BasisCache) or (
            isinstance(statement, _Scope) and statement.basis
        ):
            contraction.append(statement)
        elif isinstance(statement, (_ScalarStatement, _Initialisation, _Accumulation)) or (
            isinstance(statement, _Scope)
            and statement.index is not None
            and statement.quadrature_extent is None
        ):
            preparation.append(statement)
        else:
            return [scope]  # Leave conditionals, array-valued temporaries and nested sums intact.
    defined = {
        s.name if isinstance(s, _ScalarStatement) else s.target
        for s in preparation
        if isinstance(s, (_ScalarStatement, _Initialisation))
    }
    if any(isinstance(s, _Accumulation) and s.target not in defined for s in preparation):
        return [scope]
    uses = set(
        re.findall(
            r"[A-Za-z_][A-Za-z_0-9]*", "\n".join(replace(scope, statements=contraction).lines())
        )
    )
    live = sorted(defined & uses)
    if not live or len(live) * extent > 4096:
        return [scope]
    arrays = [f"double factor_{name}[{extent}];" for name in live]
    stores = [f"factor_{name}[{scope.index}] = {name};" for name in live]
    loads = [_ScalarStatement(name, f"factor_{name}[{scope.index}]") for name in live]
    return [
        *arrays,
        replace(scope, statements=preparation + stores),
        replace(scope, statements=loads + contraction),
    ]


def c_kernel(
    e,
    name,
    read,
    *,
    output_indices=(),
    index_bindings=None,
    unroll_indices=(),
    preintegrate=True,
    specialize_basis=True,
    reuse_transposes=True,
    sparse_tables=True,
    split_quadrature=True,
    _assign=False,
):
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
        if preintegrate and 1 <= len(output_indices) - len(unroll_indices) <= 2:
            expression = _preintegrate(
                expression,
                tuple(i for i in output_indices if i not in unroll_indices),
                share_tables=bool(unroll_indices),
            )
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
    def block(
        header,
        *,
        index=None,
        conditional=False,
        quadrature_extent=None,
        basis=False,
        output_extent=None,
    ):
        nonlocal scope
        parent = scope
        child = _Scope(
            parent,
            header,
            index,
            conditional,
            quadrature_extent=quadrature_extent,
            basis=basis,
            output_extent=output_extent,
        )
        scope = child
        try:
            yield
        finally:
            scope = parent
            # Hoisted statements have already been added to the parent.
            parent.statements.append(child)

    def temp(code):
        name = fresh("t")
        emit(_ScalarStatement(name, code))
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
            flat = len(loop_indices) > 1 and indices == tuple(loop_names.values())
            stored = values.reshape(-1) if flat else values
            shape = "".join(f"[{n}]" for n in stored.shape)
            tables[key] = (
                name,
                f"  static const {dtype} {name}{shape} = {initializer(stored)};",
                flat,
            )
        name, _, flat = tables[key]
        if flat:
            return f"{name}[{_offset(indices, values.shape)}]"
        return name + "".join(f"[{i}]" for i in indices)

    def initializer(values):
        if values.ndim == 0:
            return repr(values.item())
        return "{" + ", ".join(initializer(v) for v in values) + "}"

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
                return read(label, _offset(comp, shape))
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
                emit(_Initialisation(total))

                def reduce(indices, inner):
                    if not indices:
                        emit(_Accumulation(total, render(entry(x, inner, comp))))
                        return
                    i, *rest = indices
                    # Resolve small spatial component selections before scheduling;
                    # quadrature and DOF reductions retain their loops.
                    if not _assign and extent[i] <= 3 and i in selected_indices(x):
                        for k in range(extent[i]):
                            reduce(rest, inner | {i: k})
                    else:
                        name = fresh("i")
                        with block(
                            f"for (int {name} = 0; {name} < {extent[i]}; ++{name})",
                            index=name,
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
        if not A:
            return 1.0
        if len(A) == 1:
            return A[0][0]
        if len(A) == 2:
            return add(multiply(A[0][0], A[1][1]), negate(multiply(A[0][1], A[1][0])))
        total = 0.0
        for c, value in enumerate(A[0]):
            minor = [row[:c] + row[c + 1 :] for row in A[1:]]
            term = multiply(value, determinant(minor))
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
                emit(_BasisCache(f"double {name}" + "".join(f"[{free[i]}]" for i in axes) + ";"))
                with ExitStack() as stack:
                    inner = dict(bound)
                    for i in axes:
                        loop = fresh("p")
                        stack.enter_context(
                            block(
                                f"for (int {loop} = 0; {loop} < {free[i]}; ++{loop})",
                                index=loop,
                                basis=True,
                            )
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

    @cache
    def has_integral_reduction(n):
        return (
            isinstance(n, IndexSum) and set(loop_indices) <= dict(n.free_indices).keys()
        ) or any(has_integral_reduction(child) for child in expression_operands(n))

    def accumulate(n, bound):
        if isinstance(n, Zero):
            return
        # Integral sums surround argument loops. Other reductions remain scalar
        # computations, which entry() schedules at their earliest legal scope.
        if (
            isinstance(n, Sum)
            and loop_indices
            and set(loop_indices) <= dict(n.free_indices).keys()
            and not has_integral_reduction(n)
        ):
            emit_blocks([({}, n)], bound)
            return
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
                        block(
                            f"for (int {loop} = 0; {loop} < {extent[i]}; ++{loop})",
                            index=loop,
                            quadrature_extent=extent[i],
                        )
                    )
                    inner[i] = loop
                accumulate(n.expression, inner)
            return
        emit_blocks([({}, n)], bound)

    def emit_blocks(blocks, bound, local=None):
        if sparse_tables:
            blocks = [(selected, _sparse_contractions(n)) for selected, n in blocks]
        if len(loop_indices) in (1, 2):
            outer_indices = tuple(i for i in bound if i not in output_indices)
            # Form geometry coefficients before the basis-pair contraction when no
            # outer reduction remains. Within quadrature, coupled blocks retain
            # their shared physical derivatives to avoid duplicating preparation.
            combine_geometry = len(loop_indices) == 1 or len(blocks) == 1 or not outer_indices
            blocks = [
                (
                    selected,
                    _factor_contraction(
                        n,
                        loop_indices,
                        outer_indices,
                        combine_geometry=combine_geometry,
                        constant_basis=(
                            len(blocks) == 1
                            and len(loop_indices) == 2
                            and all(extents[i] <= 3 for i in loop_indices)
                        ),
                    ),
                )
                for selected, n in blocks
            ]
            for _, n in blocks:
                folded.update(fold(n, preserve_indices=(*output_indices, *(index_bindings or {}))))
        # Expose small reference-table entries to symbolic simplification before
        # C floating-point rules can prevent eliminating multiplication by zero.
        if (
            specialize_basis
            and local is None
            and len(loop_indices) == 2
            and all(extents[i] <= 3 for i in loop_indices)
            and prod(extents[i] for i in loop_indices) * len(blocks) <= 64
            and set(loop_indices) <= set().union(*(selected_indices(n) for _, n in blocks))
        ):
            for selected, n in blocks:
                for values in np.ndindex(tuple(extents[i] for i in loop_indices)):
                    fixed = dict(zip(loop_indices, values, strict=True))
                    specialized = simplify(substitute(n, fixed))
                    if isinstance(specialized, Zero):
                        continue
                    folded.update(fold(specialized, preserve_indices=index_bindings or {}))
                    positions = bound | selected | fixed
                    result = entry(specialized, positions, ())
                    output = _offset(
                        tuple(positions[i] for i in output_indices),
                        tuple(extents[i] for i in output_indices),
                    )
                    emit(
                        (_Assignment if _assign else _Accumulation)(f"A[{output}]", render(result))
                    )
            return
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
                        basis=True,
                        output_extent=extents[i],
                    )
                )
            for slot, (selected, n) in enumerate(blocks):
                positions = inner | selected
                result = entry(n, positions, ())
                if local is None:
                    output = _offset(
                        tuple(positions[i] for i in output_indices),
                        tuple(extents[i] for i in output_indices),
                    )
                    target = f"A[{output}]"
                else:
                    i, j = (positions[k] for k in loop_indices)
                    target = f"{local}[{slot}][{i}][{j}]"
                emit((_Assignment if _assign else _Accumulation)(target, render(result)))

    def emit_transposed_blocks(blocks, bound, axes):
        """Accumulate proven representatives, then add all component blocks to A."""
        if not (
            reuse_transposes
            and scope is root
            and len(output_indices) == 4
            and loop_indices == (output_indices[0], output_indices[2])
            and set(unroll_indices) == {output_indices[1], output_indices[3]}
            and extents[output_indices[0]] == extents[output_indices[2]]
            and extents[output_indices[1]] == extents[output_indices[3]]
            and len(blocks) == extents[output_indices[1]] ** 2
            and len(blocks) * extents[output_indices[0]] ** 2 <= 4096
            and all(not has_integral_reduction(n.expression) for _, n in blocks)
        ):
            return False
        i, r, j, s = output_indices
        representatives, references = _transpose_reuse(
            [(selected, n.expression) for selected, n in blocks], (i, j), (r, s)
        )
        if len(representatives) == len(blocks):
            return False
        if any(size == 0 for _, size in axes):
            return True
        size, components = extents[i], extents[r]
        local, row = fresh("local"), fresh("row")
        emit(f"double {local}[{len(representatives)}][{size}][{size}] = {{{{0.0}}}};")
        with ExitStack() as stack:
            inner = dict(bound)
            for axis, extent in axes:
                loop = fresh("q")
                stack.enter_context(
                    block(
                        f"for (int {loop} = 0; {loop} < {extent}; ++{loop})",
                        index=loop,
                        quadrature_extent=extent,
                    )
                )
                inner[axis] = loop
            emit_blocks(representatives, inner, local)
        slots = {
            (selected[r], selected[s]): reference
            for (selected, _), reference in zip(blocks, references, strict=True)
        }
        emit(f"double {row}[{size * components}];")
        a, b = fresh("p"), fresh("p")
        with block(f"for (int {a} = 0; {a} < {size}; ++{a})", index=a):
            for component in range(components):
                with block(f"for (int {b} = 0; {b} < {size}; ++{b})", index=b):
                    for c in range(components):
                        slot, transposed = slots[component, c]
                        first, second = (b, a) if transposed else (a, b)
                        emit(
                            f"{row}[{b} * {components} + {c}] = {local}[{slot}][{first}][{second}];"
                        )
                column = fresh("p")
                with block(
                    f"for (int {column} = 0; {column} < {size * components}; ++{column})",
                    index=column,
                ):
                    target = (
                        f"A[({a} * {components} + {component}) * {size * components} + {column}]"
                    )
                    emit(_Accumulation(target, f"{row}[{column}]"))
        return True

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
                if emit_transposed_blocks(blocks, bound, axes):
                    return
                with ExitStack() as stack:
                    inner = dict(bound)
                    for i, size in axes:
                        if size == 0:
                            return
                        loop = fresh("q")
                        stack.enter_context(
                            block(
                                f"for (int {loop} = 0; {loop} < {size}; ++{loop})",
                                index=loop,
                                quadrature_extent=size,
                            )
                        )
                        inner[i] = loop
                    accumulate_blocks([(selected, n.expression) for selected, n in blocks], inner)
                return
        if all(
            not isinstance(n, (Sum, IndexSum)) or (loop_indices and not has_integral_reduction(n))
            for _, n in blocks
        ):
            emit_blocks(blocks, bound)
        else:
            for selected, n in blocks:
                accumulate(n, bound | selected)

    # Reusing loop names also lets prepared basis arrays match across component blocks.
    loop_names = {i: fresh("a") for i in loop_indices}
    bound = {i: 0 for i, _ in e.free_indices} | (index_bindings or {})
    if all(dict(e.free_indices)[i] > 0 for i in output_indices):
        if _assign:
            # Private stage outputs are evaluated one entry at a time. A reduction
            # becomes a local scalar accumulator inside these output loops.
            emit_blocks(expressions, bound)
        else:
            accumulate_blocks(expressions, bound)
    root.inline_products()
    if split_quadrature:
        [root] = _split_quadrature(root)
    root = _flatten_output(root)
    body = "\n".join([declaration for _, declaration, _ in tables.values()] + list(root.lines()))
    return (
        "#include <stdint.h>\n"
        f"void {name}(double* restrict A, const double* restrict w, const double* restrict c,\n"
        "    const double* restrict coordinate_dofs, const int* restrict entity_local_index,\n"
        "    const uint8_t* restrict quadrature_permutation, void* custom_data)\n"
        f"{{\n{body}\n}}\n"
    )


def render(x):
    return repr(x) if isinstance(x, float) else x
