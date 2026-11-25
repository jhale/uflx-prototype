# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Factor a fixed quadrature sum into reference tensors and cell coefficients."""

from collections import defaultdict
from functools import cache, lru_cache, reduce
from itertools import product
from math import prod
from operator import add, attrgetter, mul

import jax
import jax.numpy as jnp
import numpy as np

from uflx.algorithms import operands
from uflx.arithmetic import num
from uflx.nodes import (
    Absolute,
    AbstractExpression,
    AbstractTopology,
    CellMap,
    ComponentTensor,
    Coordinate,
    Determinant,
    Division,
    Float,
    Form,
    Function,
    Gradient,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Inverse,
    Logarithm,
    Power,
    Product,
    PullBack,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)
from uflx.tensors import indexed, tensor

from ._basis_tables import literal_values, sampled_quadrature
from ._finite_elements import basix_space, is_basix_basis

# Bound distributive expansion and the largest table (including its quadrature axis).
_MAX_EXPANSION = 256
_MAX_ENTRIES = 128 * 1024


_ALGEBRA = (
    Absolute,
    ComponentTensor,
    Coordinate,
    Determinant,
    Division,
    Float,
    Gradient,
    Indexed,
    IndexSum,
    Integer,
    Inverse,
    Logarithm,
    Power,
    Product,
    PullBack,
    SquareRoot,
    Sum,
    Tensor,
    Zero,
)


class _UseQuadrature(Exception):
    """The ordinary quadrature kernel handles this expression."""


def _sampled(body, inputs):
    """Recognize the actual weighted point evaluations, independently of index labels."""
    sampled = sampled_quadrature(body, inputs)
    if sampled is not None:
        expr = sampled[0]
        if not (expr.free_indices or expr.argument_spaces or expr.value_shape):
            return sampled
    raise _UseQuadrature


def _factors(body, arguments, inputs):
    """Expand bound component indices and collect scalar products with equal cell factors."""

    @cache
    def fields(node):
        if isinstance(node, Function):
            if node.scalar_type != "real" or (node not in arguments and node not in inputs):
                raise _UseQuadrature
            return frozenset((node,))
        if is_basix_basis(node):
            return frozenset()
        if isinstance(node, AbstractExpression) and (
            node.scalar_type != "real" or not isinstance(node, _ALGEBRA)
        ):
            raise _UseQuadrature
        names = [name for name in node.argument_names if name not in node.labels]
        return frozenset().union(*(fields(child) for child in operands(node, names)))

    # Unknown operators and lexical binders must keep their original sampling semantics.
    used = fields(body)
    runtime = tuple(x for x in inputs if x in used and x not in arguments)

    visits = 0

    def expand(node, components, bound):
        nonlocal visits
        visits += 1
        if visits > _MAX_EXPANSION:
            raise _UseQuadrature
        deps = fields(node)
        if not node.free_indices and deps <= arguments:
            yield (indexed(node, components),), ()
            return
        match node:
            case Indexed(expr, indices):
                try:
                    selected = tuple(i if isinstance(i, int) else bound[i] for i in indices)
                except KeyError:
                    raise _UseQuadrature from None
                yield from expand(expr, selected, bound)
            case ComponentTensor(expr, indices):
                yield from expand(expr, (), bound | dict(zip(indices, components, strict=True)))
            case IndexSum(expr, indices):
                extents = dict(expr.free_indices)
                sizes = [extents[i] for i in indices]
                if (
                    not all(isinstance(n, int) and n > 0 for n in sizes)
                    or prod(sizes) > _MAX_EXPANSION
                ):
                    raise _UseQuadrature
                for key in product(*(range(n) for n in sizes)):
                    yield from expand(
                        expr, components, bound | dict(zip(indices, key, strict=True))
                    )
            case Product(a, b):
                for r, c in expand(a, components if a.value_shape else (), bound):
                    for s, d in expand(b, components if b.value_shape else (), bound):
                        yield r + s, c + d
            case Sum(a, b):
                yield from expand(a, components, bound)
                yield from expand(b, components, bound)
            case Tensor(entries):
                yield from expand(entries[components[0]], components[1:], bound)
            case _ if not node.free_indices and deps.isdisjoint(arguments):
                yield (), (indexed(node, components),)
            case _:
                raise _UseQuadrature

    # R_ijrs and R_ijsr can share a cell coefficient. Sorting only scalar factors
    # also preserves the distinct coefficients of a nonsymmetric form.
    by_reference = defaultdict(list)
    digest = attrgetter("digest")
    for reference, cell in expand(body, (), {}):
        by_reference[tuple(sorted(reference, key=digest))].append(tuple(sorted(cell, key=digest)))

    def monomial(factors):
        return reduce(mul, factors) if factors else num(1)

    by_cell = defaultdict(list)
    for factors, coefficients in by_reference.items():
        polynomial = tuple(sorted(coefficients, key=lambda term: tuple(map(digest, term))))
        by_cell[polynomial].append(factors)
    reference = tensor(tuple(reduce(add, map(monomial, terms)) for terms in by_cell.values()))
    return reference, by_cell, runtime


def _basix_reference_table(groups, arguments, cell, points, weights, dtype):
    """Integrate products of scalar basis values and first partials at fixed points."""
    import basix

    shape = (len(weights), *(arg.space.num_dofs for arg in arguments))
    points = np.asarray(points, dtype=dtype)
    weights = np.asarray(weights, dtype=dtype)

    @cache
    def tabulate(space):
        element = space.element(0)
        table = element.tabulate(1, points.astype(element.dtype, copy=False))
        return table[..., 0].astype(dtype, copy=False)

    samples = {}
    for axis, arg in enumerate(arguments):
        space = arg.space
        domain = space.domain
        if space.value_shape:
            return None
        if domain == cell:
            field = arg
        elif isinstance(domain, AbstractTopology) and domain.cell_types == (cell,):
            field = PullBack(arg, CellMap(domain, 0))
        else:
            return None
        element = space.element(0)
        if element.family != basix.ElementFamily.P or element.cell_type not in (
            basix.CellType.interval,
            basix.CellType.triangle,
            basix.CellType.tetrahedron,
        ):
            return None
        gradient = Gradient(field, cell)
        # Basix orders its rows as the value followed by the coordinate partials.
        factors = (field, *(indexed(gradient, (i,)) for i in range(cell.dimension)))
        dofs = [1] * len(arguments)
        dofs[axis] = element.dim
        for factor, values in zip(factors, tabulate(space), strict=True):
            samples[factor] = values.reshape(len(weights), *dofs)

    rows = []
    for terms in groups:
        try:
            values = sum(prod(samples[factor] for factor in term) for term in terms)
        except KeyError:
            return None
        values = np.broadcast_to(np.asarray(values, dtype=dtype), shape)
        rows.append(np.tensordot(weights, values, axes=1))
    return np.stack(rows)


def _contraction(body, arguments, inputs, ordinary):
    expr, cell, points, weights = _sampled(body, inputs)
    basis_size = prod(arg.space.num_dofs for arg in arguments)
    if not all(isinstance(n, int) for n in points.value_shape) or (
        basis_size * prod(points.value_shape) > _MAX_ENTRIES
    ):
        raise _UseQuadrature
    reference, coefficients, runtime = _factors(expr, frozenset(arguments), inputs)
    if basis_size * weights.value_shape[0] * reference.value_shape[0] > _MAX_ENTRIES:
        raise _UseQuadrature
    try:
        points, weights = np.asarray(literal_values(points)), np.asarray(literal_values(weights))
    except ValueError:
        raise _UseQuadrature from None
    basis = ordinary(Form(reference, arguments), (cell,))
    # Reuse cell factors shared across different entries of the geometry tensor.
    leaves = dict.fromkeys(f for terms in coefficients for term in terms for f in term)
    reads = {f: ordinary(f, (*runtime, cell)) for f in leaves}

    def read(*args):
        values = {f: evaluate(*args) for f, evaluate in reads.items()}
        return jnp.stack(
            [
                sum(prod(values[f] for f in term) for term in polynomial)
                for polynomial in coefficients
            ]
        )

    positions = [inputs.index(x) for x in runtime]
    numeric_kinds = ["fiu" if basix_space(x.space) is not None else "f" for x in runtime]

    @lru_cache(maxsize=2)
    def reference_table(dtype):
        # Store a host constant, never a tracer from an enclosing jit, vmap or AD.
        table = _basix_reference_table(
            coefficients.values(), arguments, cell, points, weights, dtype
        )
        if table is not None:
            return table
        with jax.ensure_compile_time_eval():
            samples = jax.jit(jax.vmap(basis))(jnp.asarray(points, dtype=dtype))
            table = jnp.tensordot(jnp.asarray(weights, dtype=dtype), samples, axes=1)
            return np.ascontiguousarray(np.moveaxis(np.asarray(table), -1, 0))

    @lru_cache(maxsize=4)
    def table_for(specs, dtype):
        # A traced result may have out_axes=None only if it carries no point axis.
        # Abstract coefficient values keep this proof independent of the current cell.
        mapped = jax.vmap(read, in_axes=(None,) * len(specs) + (0,), out_axes=None)
        try:
            jax.eval_shape(mapped, *specs, jax.ShapeDtypeStruct(points.shape, dtype))
            return reference_table(dtype)
        except (KeyError, TypeError, ValueError):
            # A point-dependent result or another required binding uses ordinary evaluation.
            return None

    fallback = ordinary(Form(body, arguments), inputs)

    def kernel(*values):
        if len(values) != len(inputs):
            return fallback(*values)
        data = tuple(values[i] for i in positions)
        if any(callable(v) or isinstance(v, (dict, AbstractExpression)) for v in data):
            return fallback(*values)
        try:
            arrays = tuple(map(jnp.asarray, data))
        except (TypeError, ValueError):
            return fallback(*values)
        if all(v.dtype.kind in kinds for v, kinds in zip(arrays, numeric_kinds, strict=True)):
            specs = tuple(jax.ShapeDtypeStruct(v.shape, v.dtype) for v in arrays)
            table = table_for(specs, np.float64 if jax.config.x64_enabled else np.float32)
            if table is not None:
                return jnp.tensordot(read(*arrays, points[0]), table, axes=1)
        return fallback(*values)

    return kernel


def reference_tensor_kernel(expression, inputs, ordinary):
    """Specialize eligible integrals independently, retaining the usual trace elsewhere."""
    if not isinstance(expression, (Form, Sum)) or any(
        not (
            isinstance(key, (Function, Integral))
            or (isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], Integral))
        )
        for key in inputs
    ):
        return None

    kernels = {}
    sums = {}

    @cache
    def visit(expr):
        if expr.free_indices or expr.value_shape or expr.scalar_type != "real":
            return False
        if isinstance(expr, Form):
            args, body = expr.arguments, expr.expression
            if (
                not args
                or body.argument_spaces
                or any(
                    not isinstance(a, Function)
                    or basix_space(a.space) is None
                    or a.scalar_type != "real"
                    for a in args
                )
            ):
                return False
            if not isinstance(body, Sum):
                try:
                    kernels[expr] = _contraction(body, args, inputs, ordinary)
                except _UseQuadrature:
                    return False
                return True
            a, b = Form(body.first, args), Form(body.second, args)
        elif isinstance(expr, Sum):
            a, b = expr.first, expr.second
        else:
            return False
        left, right = visit(a), visit(b)
        if not (left or right):
            return False
        for term in (a, b):
            if term not in sums and term not in kernels:
                kernels[term] = ordinary(term, inputs)
        sums[expr] = (a, b)
        return True

    if not visit(expression):
        return None
    if not sums:
        return kernels[expression]

    def kernel(*values):
        results = {expr: compute(*values) for expr, compute in kernels.items()}
        # visit inserted the sums after their operands.
        for expr, (left, right) in sums.items():
            results[expr] = results[left] + results[right]
        return results[expression]

    return kernel
