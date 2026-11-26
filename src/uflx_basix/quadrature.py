# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Symbolic quadrature on Basix reference cells."""

from itertools import product
from math import comb, prod

import basix

from uflx import nodes
from uflx.algorithms import (
    argument_leaves,
    complete_family,
    derived_index,
    derived_indices,
    expression_operands,
    occurs,
    promote,
    substitute_values,
)
from uflx.arithmetic import num
from uflx.geometry import pull_back
from uflx.nodes import Form, Integral, Zero
from uflx.tensors import component_tensor, index_sum, indexed, tensor
from uflx_extensions.runtime_arrays import runtime_array

from .evaluation import _point_map, _real_array
from .evaluation import evaluate_at_points as evaluate_at_points
from .nodes import BasixSpace


def _argument_uses(body, fields, cell):
    """Match whole derivatives before their operands, only at the integration point."""
    seen = set()
    pending = [body]
    while pending:
        node = pending.pop()
        if node in seen:
            continue
        seen.add(node)
        if node.is_pointwise:
            pending.extend(reversed(expression_operands(node)))
            continue
        leaf = node
        order = 0
        while isinstance(leaf, nodes.Gradient) and leaf.domain == cell:
            leaf = leaf.expression
            order += 1
        slot = 0
        if (
            isinstance(leaf, nodes.PullBack)
            and isinstance(leaf.map, nodes.CellMap)
            and isinstance(leaf.map.parent, nodes.AbstractTopology)
            and leaf.map.reference_cell == cell
        ):
            slot, leaf = leaf.map.cell_slot, leaf.expression
        if leaf in fields and node.domain == cell:
            yield node, leaf, slot, order
        elif any(occurs(f, node) for f in fields):
            raise ValueError(
                "form argument must be evaluated at the reference sampling point", node
            )


def _basis_value(operand, q, axis, space, slot, offset, width, order, dim):
    element = space.element(slot)
    shape = (*space.value_shape, *(dim,) * order)
    # Basix stores each mixed partial once, including for vector-valued elements.
    partials = [
        basix.index(*(axes.count(d) for d in range(dim))) if order else 0
        for axes in product(range(dim), repeat=order)
    ]

    def column(dof, block=None):
        values = [
            operand[row, q, dof, c if block is None else 0]
            if block is None or block == c
            else num(0)
            for c in range(prod(space.value_shape))
            for row in partials
        ]
        for extent in reversed(shape):
            values = [tensor(tuple(values[i : i + extent])) for i in range(0, len(values), extent)]
        return values[0]

    if width == element.dim and not space.block_shape:
        return column(axis)
    # Product members and cell slots occupy contiguous ranges. A blocked
    # element interleaves its components within each native basis column.
    values = [nodes.Zero(shape, (), (), nodes.ScalarType.real)] * width
    for local, packed in enumerate(space.cell_dofs[slot]):
        dof, block = divmod(local, space.block_size)
        values[offset + packed] = column(dof, block if space.block_shape else None)
    return tensor(tuple(values))[axis]


def _tabulate_arguments(integral, arguments, points, q):
    body = integral.expression
    cell = integral.integration_domain
    axes = derived_indices("uflx_basix.form_dof", (integral, arguments, points, q), len(arguments))
    fields = {}
    widths = {}
    for axis, entry in zip(axes, arguments, strict=True):
        offset = 0
        for member in argument_leaves(entry):
            f = member.operand if isinstance(member, nodes.Conjugate) else member
            space = f.space.space if isinstance(f.space, nodes.ConjugateSpace) else f.space
            if not isinstance(space, BasixSpace):
                raise TypeError("sample_form requires Basix form arguments")
            fields[f] = (space, axis, offset)
            offset += space.num_dofs
        widths[axis] = offset

    uses = {}
    replacements = {}
    orders = {}
    for node, f, slot, order in _argument_uses(body, fields, cell):
        if 0 in node.value_shape:
            replacements[node] = nodes.Zero(node.value_shape, (), (), node.scalar_type)
            continue
        uses[node] = (f, slot, order)
        data = fields[f][0].element_data[slot]
        orders[data] = max(orders.get(data, 0), order)
    tables = {
        (data.native, order): runtime_array(
            f"basis_{data.digest:032x}_{points.digest:032x}_{order}",
            (
                comb(cell.dimension + order, order),
                points.value_shape[0],
                data.native.dim,
                data.native.value_size,
            ),
        )
        for data, order in orders.items()
    }
    for node, (f, slot, order) in uses.items():
        space, axis, offset = fields[f]
        data = space.element_data[slot]
        operand = tables[data.native, orders[data]]
        value = _basis_value(
            operand, q, axis, space, slot, offset, widths[axis], order, cell.dimension
        )
        replacements[node] = promote(node, value)
    result = substitute_values(body, replacements)
    # Even an unused form slot contributes its local tensor axis.
    result = complete_family(tuple((dict(result.free_indices) | widths).items()), result)
    components = derived_indices(
        "uflx_basix.form_component", (integral, arguments, points, q), len(body.value_shape)
    )
    return component_tensor(indexed(result, components), axes + components), tables


def sample_form(form, points, q):
    """Sample a form over one lowered Basix reference-cell integral, leaving q free.

    The integral supplies the integrand and cell. Each declared form argument
    contributes a leading local DOF axis, in declaration order.
    The second result maps (native element, highest derivative order) to the
    runtime_array used for its form arguments. Supply element.tabulate(order, points)
    directly. Form arguments must be evaluated at the integration point.
    Coefficients and geometry retain their sampling expressions.
    """
    if not isinstance(form, Form) or not isinstance(form.expression, Integral):
        raise TypeError("expected a Form over a single reference-cell Integral")
    if form.expression.argument_spaces:
        raise ValueError("declare the form arguments on the outer Form")
    points = _real_array(points)
    mapping = _point_map(form.expression.integration_domain, points, q)
    integrand, tables = _tabulate_arguments(form.expression, form.arguments, points, q)
    return Integral(pull_back(integrand, mapping), mapping.parameter_domain), tables


def weighted_sum(samples, weights, q):
    """Sum weights[q] times samples over q, preserving all other axes."""
    weights = _real_array(weights)
    if weights.value_shape != (dict(samples.free_indices).get(q),):
        raise ValueError("expected weights of shape (n,) matching the samples' point index")
    axes = derived_indices("uflx_basix.quadrature_component", (samples,), len(samples.value_shape))
    total = component_tensor(index_sum(weights[q] * indexed(samples, axes), (q,)), axes)
    if weights.value_shape == (0,):
        return Zero(total.value_shape, total.free_indices, total.argument_spaces, total.scalar_type)
    return total


def apply_quadrature(integral, points, weights):
    """Evaluate one reference-cell integral by sampling its integrand and summing with weights."""
    if not isinstance(integral, Integral):
        raise TypeError("expected a single Integral")
    q = derived_index("uflx_basix.quadrature", integral)
    samples = evaluate_at_points(integral.expression, points, q, cell=integral.integration_domain)
    return weighted_sum(samples, weights, q)
