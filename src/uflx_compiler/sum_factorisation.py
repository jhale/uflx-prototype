"""Staged contractions of scalar tensor-product expressions."""

from collections.abc import Callable, Sequence
from functools import cache, reduce
from itertools import product
from math import prod

import numpy as np

from uflx.algorithms import derived_index, expression_operands, map_fields, replace
from uflx.arithmetic import add, divide, multiply, num
from uflx.nodes import (
    AbstractExpression,
    BroadcastTo,
    Division,
    Index,
    Indexed,
    IndexSum,
    Product,
    Sum,
    Tensor,
)
from uflx.tensors import broadcast_to, index_sum, tensor
from uflx_basix.nodes import RuntimeArray
from uflx_basix.quadrature import constant
from uflx_basix.runtime import runtime_array

from .c_code import c_kernel
from .evaluation import evaluate
from .inputs import runtime_arrays
from .simplification import is_zero, simplify


def _product(factors):
    return reduce(multiply, factors, num(1))


def _factors(expression):
    # Index completion can wrap a separable scalar product in a broadcast when a
    # derivative is constant along one direction. Keep those repeated axes as
    # independent unit factors, rather than hiding the whole product from planning.
    if (
        isinstance(expression, Indexed)
        and isinstance(expression.expression, BroadcastTo)
        and not expression.expression.value.value_shape
    ):
        value = expression.expression.value
        repeated = tuple(
            broadcast_to(num(1), (n,))[i]
            for i, n in expression.free_indices
            if i not in dict(value.free_indices)
        )
        return (*_factors(value), *repeated)
    if isinstance(expression, Product) and not expression.value_shape:
        return (*_factors(expression.first), *_factors(expression.second))
    if isinstance(expression, Division):
        return (*_factors(expression.numerator), divide(num(1), expression.denominator))
    return (expression,)


def _terms(expression, outputs):
    """Distribute only sums involving test functions; keep pointwise work intact."""
    if not outputs.intersection(i for i, _ in expression.free_indices):
        return (expression,)
    if isinstance(expression, Sum):
        return (*_terms(expression.first, outputs), *_terms(expression.second, outputs))
    if isinstance(expression, Product):
        return tuple(
            multiply(a, b)
            for a in _terms(expression.first, outputs)
            for b in _terms(expression.second, outputs)
        )
    if isinstance(expression, Division) and not outputs.intersection(
        i for i, _ in expression.denominator.free_indices
    ):
        return tuple(
            divide(a, expression.denominator) for a in _terms(expression.numerator, outputs)
        )
    return (expression,)


def _constant_product(factors):
    """Tabulate the small constant part of a directional contraction."""
    constants = tuple(f for f in factors if not runtime_arrays(f))
    varying = tuple(f for f in factors if runtime_arrays(f))
    if len(constants) < 2:
        return _product(factors)
    expression = _product(constants)
    if len(expression.free_indices) > 2 or prod(n for _, n in expression.free_indices) > 4096:
        return _product(factors)
    table = constant(evaluate(expression, {}, np))
    axes = tuple(i for i, _ in expression.free_indices)
    return multiply(_product(varying), table[axes] if axes else table)


def _contract(expression, indices):
    """Eliminate each axis while keeping independent factors outside its sum."""
    factors = _factors(expression)
    for index in reversed(indices):
        dependent = tuple(f for f in factors if index in dict(f.free_indices))
        independent = tuple(f for f in factors if index not in dict(f.free_indices))
        factors = (*independent, index_sum(_constant_product(dependent), (index,)))
    return _product(factors)


def action_kernel(
    expression: AbstractExpression,
    name: str,
    read: Callable[[object, int | str], str],
    output_indices: tuple[Index, ...],
    permutation: Sequence[int],
) -> str | None:
    """Emit bounded intermediate arrays and retain the external DOF ordering.

    Returns None if the explicit staged arrays exceed 256 KiB. This initial path is
    for scalar actions; matrix and facet scheduling use the ordinary emitter.
    """
    outputs = set(output_indices)

    @cache
    def components(node):
        own = (
            {i for i in node.indices if isinstance(i, Index)}
            if isinstance(node, Indexed)
            and isinstance(node.expression, Tensor)
            and runtime_arrays(node.expression)
            else set()
        )
        return own.union(*(components(child) for child in expression_operands(node)))

    @cache
    def factor(node):
        node = map_fields(node, factor, node.value_fields)
        if isinstance(node, IndexSum):
            selected = tuple(i for i in node.indices if i in components(node.expression))
            extents = dict(node.expression.free_indices)
            if selected and all(extents[i] <= 3 for i in selected):
                remaining = tuple(i for i in node.indices if i not in selected)
                expanded = reduce(
                    add,
                    (
                        factor(
                            simplify(
                                replace(node.expression, dict(zip(selected, values, strict=True)))
                            )
                        )
                        for values in product(*(range(extents[i]) for i in selected))
                    ),
                    num(0),
                )
                return factor(index_sum(expanded, remaining)) if remaining else expanded
            terms = _terms(node.expression, outputs)
            if outputs.intersection(i for i, _ in node.free_indices):
                groups = {}
                for term in terms:
                    factors = _factors(term)
                    basis = tuple(
                        f
                        for f in factors
                        if outputs.intersection(i for i, _ in f.free_indices)
                        or (
                            len(f.free_indices) == 1
                            and set(dict(f.free_indices)) <= set(node.indices)
                            and not runtime_arrays(f)
                        )
                    )
                    coefficient = _product(f for f in factors if f not in basis)
                    groups[basis] = (
                        add(groups[basis], coefficient) if basis in groups else coefficient
                    )
                channel = derived_index("compiler.sum_factor.channel", node)
                groups = {basis: simplify(value) for basis, value in groups.items()}
                groups = {basis: value for basis, value in groups.items() if not is_zero(value)}
                if not groups:
                    return num(0)
                source = simplify(tensor(tuple(groups.values()))[channel])
                axes = (*(i for i, _ in source.free_indices if i != channel), channel)
                shape = tuple(dict(source.free_indices)[i] for i in axes)
                values = runtime_array(("pointwise", source, axes, channel), shape)
                terms = tuple(
                    multiply(values[(*axes[:-1], k)], _product(basis))
                    for k, basis in enumerate(groups)
                )
            return reduce(add, (_contract(term, node.indices) for term in terms), num(0))
        return node

    stages = []
    sizes = []

    def store(node, axes, unroll=()):
        label = len(stages)
        shape = tuple(dict(node.free_indices)[i] for i in axes)
        array = runtime_array(("sum_factor", label), shape)
        stages.append((node, tuple(axes), label, unroll))
        sizes.append(prod(shape))
        return array

    @cache
    def materialize(node):
        if (
            isinstance(node, RuntimeArray)
            and isinstance(node.label, tuple)
            and node.label[0] == "pointwise"
        ):
            _, source, axes, channel = node.label
            return store(materialize(source), axes, (channel,))
        node = map_fields(node, materialize, node.value_fields)
        if not isinstance(node, IndexSum):
            return node
        axes = tuple(i for i, _ in node.free_indices)
        array = store(node, axes)
        return array[tuple(axes)] if axes else array

    result = materialize(simplify(factor(expression)))
    if sum(sizes) + len(permutation) > 32768:
        return None

    def access(label, offset):
        if isinstance(label, tuple) and label[0] == "sum_factor":
            return f"sf{label[1]}[{offset}]"
        return read(label, offset)

    def body(node, axes, target, unroll=()):
        source = c_kernel(
            node,
            name,
            access,
            output_indices=axes,
            unroll_indices=unroll,
            preintegrate=False,
            specialize_basis=False,
            reuse_transposes=False,
            sparse_tables=False,
            split_quadrature=False,
            _assign=True,
        )
        return source.split("{\n", 1)[1].rsplit("}", 1)[0].replace("A[", target + "[")

    parts = []
    for (node, axes, label, unroll), size in zip(stages, sizes, strict=True):
        parts.append(f"  double sf{label}[{size}];")
        parts.append("  {\n" + body(node, axes, f"sf{label}", unroll) + "  }")
    count = len(permutation)
    parts.append(f"  double result[{count}];")
    parts.append("  {\n" + body(result, output_indices, "result") + "  }")
    values = ", ".join(map(str, permutation))
    parts.append(f"  static const int dofs[{count}] = {{{values}}};")
    parts.append(f"  for (int i = 0; i < {count}; ++i) A[dofs[i]] += result[i];")
    return (
        "#include <stdint.h>\n"
        f"void {name}(double* restrict A, const double* restrict w, const double* restrict c,\n"
        "    const double* restrict coordinate_dofs, const int* restrict entity_local_index,\n"
        "    const uint8_t* restrict quadrature_permutation, void* custom_data)\n"
        "{\n" + "\n".join(parts) + "\n}\n"
    )
