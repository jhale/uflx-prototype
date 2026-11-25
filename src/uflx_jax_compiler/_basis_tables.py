# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Array inputs for finite-element evaluations at known reference quadrature points."""

from dataclasses import dataclass
from math import comb

import jax.numpy as jnp
import numpy as np

from uflx import nodes
from uflx.algorithms import expression_operands
from uflx.tensors import indexed

from . import _finite_elements as elements


@dataclass(frozen=True)
class BasisTable:
    """A Basix tabulation for an explicit points operand or an integral with a runtime rule."""

    point_source: nodes.AbstractExpression
    _element_data: nodes.Base
    order: int

    @property
    def element(self):
        return self._element_data.native


def keeps_quadrature_point(node):
    return (
        node.is_pointwise
        or isinstance(node, (nodes.Gradient, nodes.Divergence, nodes.Curl))
        or (
            isinstance(node, nodes.PullBack)
            and isinstance(node.map, nodes.CellMap)
            and isinstance(node.map.parent, nodes.AbstractTopology)
        )
    )


def sampled_points(mapping):
    """Identify the reference cell and row selected by evaluate_at_points."""
    match mapping:
        case nodes.DomainMap(
            nodes.EuclideanSpace(0),
            nodes.ComponentTensor(nodes.Indexed(points, (q, j)), (k,)),
            cell,
        ) if (
            j == k
            and q != j
            and isinstance(q, nodes.Index)
            and isinstance(cell, nodes.AbstractReferenceCell)
            and points.domain is None
            and not points.argument_spaces
            and not points.free_indices
            and points.scalar_type == nodes.ScalarType.real
        ):
            return points, q, cell
    return None


def sampled_quadrature(node, bindings=()):
    """Read the weighted reference-point sum by its algebra, independently of index labels."""
    match node:
        case nodes.IndexSum(nodes.Product(nodes.Indexed(weights, (w,)), sample), (q,)) if w == q:
            components = ()
            if isinstance(sample, nodes.Indexed):
                sample, components = sample.expression, sample.indices
            match sample:
                case nodes.Integral(
                    nodes.PullBack(expr, mapping), nodes.EuclideanSpace(0) as domain
                ):
                    source = sampled_points(mapping)
                    if (
                        source is not None
                        and source[1] == q
                        and weights.domain is None
                        and not weights.argument_spaces
                        and not weights.free_indices
                        and weights.scalar_type == nodes.ScalarType.real
                        and sample not in bindings
                        and (sample, domain) not in bindings
                    ):
                        points, _, cell = source
                        return (
                            indexed(expr, components) if components else expr,
                            cell,
                            points,
                            weights,
                        )
    return None


def basis_tables(expression):
    """Return the optional basis-table inputs of a reference-cell form in traversal order.

    Each request identifies the source of the points, a native Basix element
    and the highest reference derivative needed there. Pass the desired requests to
    ``uflx_jax_compiler.trace``.
    """
    orders = {}
    # Outside an integral, locate integral nodes. Inside, follow only paths
    # known to keep the reference quadrature coordinate.
    pending = [(expression, None, None, 0, 0)]
    seen = set()
    while pending:
        state = pending.pop()
        if state in seen:
            continue
        seen.add(state)
        node, point_source, domain, slot, order = state
        sampled = sampled_quadrature(node)
        if sampled is not None:
            expr, cell, points, weights = sampled
            pending.extend(
                (
                    (weights, None, None, 0, 0),
                    (points, None, None, 0, 0),
                    (expr, points, cell, 0, 0),
                )
            )
            continue
        if isinstance(node, nodes.PullBack) and (source := sampled_points(node.map)) is not None:
            points, _, cell = source
            pending.extend(((points, None, None, 0, 0), (node.expression, points, cell, 0, 0)))
            continue
        if isinstance(node, nodes.Integral):
            domain, slot, order = node.integration_domain, 0, 0
            point_source = node if isinstance(domain, nodes.AbstractReferenceCell) else None
        elif point_source is not None:
            if isinstance(node, nodes.Function) and node.domain == domain:
                space = elements.basix_space(node.space)
                if space is not None:
                    key = (point_source, space.element_data[slot])
                    orders[key] = max(order, orders.get(key, 0))
                continue
            if elements.is_basix_basis(node):
                if node.domain == domain and node.cell_slot == slot:
                    key = (
                        point_source,
                        node.space.element_data[0 if node.space.cell is not None else slot],
                    )
                    orders[key] = max(order, orders.get(key, 0))
                continue
            if not keeps_quadrature_point(node) or (
                isinstance(node, nodes.PullBack) and node.map.reference_cell != domain
            ):
                # Still discover any independent, nested integrals.
                point_source, domain, slot, order = None, None, 0, 0
            elif isinstance(node, nodes.PullBack):
                domain, slot = node.map.parent, node.map.cell_slot
            elif (
                isinstance(node, (nodes.Gradient, nodes.Divergence, nodes.Curl))
                and node.domain == domain
            ):
                order += 1
        pending.extend(
            (child, point_source, domain, slot, order)
            for child in reversed(expression_operands(node))
        )
    return tuple(
        BasisTable(point_source, data, order) for (point_source, data), order in orders.items()
    )


def literal_values(node):
    match node:
        case nodes.Float(x) | nodes.Integer(x):
            return x
        case nodes.Tensor(entries):
            return tuple(map(literal_values, entries))
        case nodes.Zero() if (
            not node.free_indices
            and not node.argument_spaces
            and all(isinstance(n, int) for n in node.value_shape)
        ):
            return np.zeros(node.value_shape)
    raise ValueError("expected a literal array")


def fixed_basis_tables(expression, inputs):
    tables = {}
    for request in basis_tables(expression):
        if request in inputs or request.point_source in inputs:
            continue
        try:
            points = literal_values(request.point_source)
        except ValueError:
            continue
        element = request.element
        tables[request] = element.tabulate(request.order, np.asarray(points, dtype=element.dtype))
    return tables


def quadrature_tables(point_source, cell, bindings, npoints):
    """Move the point axis to the front for vmap over this integral's samples."""
    tables = {}
    for key, values in bindings.items():
        if isinstance(key, BasisTable) and key.point_source == point_source:
            element = key.element
            dim = cell.dimension
            shape = (comb(dim + key.order, key.order), npoints, element.dim, element.value_size)
            array = jnp.asarray(values)
            if array.shape != shape or not jnp.issubdtype(array.dtype, jnp.floating):
                raise ValueError(
                    f"expected a floating Basix table with shape {shape}, got {array.shape}"
                )
            tables[key._element_data] = jnp.swapaxes(array, 0, 1)
    return tables
