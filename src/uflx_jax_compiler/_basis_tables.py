# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Array inputs for finite-element evaluations at known reference quadrature points."""

from dataclasses import dataclass
from math import comb

import jax.numpy as jnp

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


# These operators evaluate their value operands at the current point. An external
# operator can instead take a whole field and choose where to evaluate it.
_POINTWISE = (
    nodes.Absolute,
    nodes.BroadcastTo,
    nodes.ComponentTensor,
    nodes.Conditional,
    nodes.Conjugate,
    nodes.Determinant,
    nodes.Division,
    nodes.ImaginaryPart,
    nodes.Indexed,
    nodes.IndexSum,
    nodes.Inverse,
    nodes.LessThan,
    nodes.Logarithm,
    nodes.Power,
    nodes.Product,
    nodes.RealPart,
    nodes.SquareRoot,
    nodes.Sum,
    nodes.Tensor,
    nodes.Gradient,
    nodes.Divergence,
    nodes.Curl,
)


def keeps_quadrature_point(node):
    return isinstance(node, _POINTWISE) or (
        isinstance(node, nodes.PullBack)
        and isinstance(node.map, nodes.CellMap)
        and isinstance(node.map.parent, nodes.AbstractTopology)
    )


def sampled_quadrature(node, bindings=()):
    """Read the weighted reference-point sum by its algebra, independently of index labels."""
    match node:
        case nodes.IndexSum(nodes.Product(nodes.Indexed(weights, (w,)), sample), (q,)):
            components = ()
            if isinstance(sample, nodes.Indexed):
                sample, components = sample.expression, sample.indices
            match sample:
                case nodes.Integral(
                    nodes.PullBack(
                        expr,
                        nodes.DomainMap(
                            source, nodes.ComponentTensor(nodes.Indexed(points, (p, j)), (k,)), cell
                        ),
                    ),
                    domain,
                ):
                    if (
                        q == w == p
                        and j == k
                        and q != j
                        and domain == source == nodes.EuclideanSpace(0)
                        and isinstance(cell, nodes.AbstractReferenceCell)
                        and all(
                            x.domain is None
                            and not x.argument_spaces
                            and not x.free_indices
                            and x.scalar_type == nodes.ScalarType.real
                            for x in (points, weights)
                        )
                        and sample not in bindings
                        and (sample, domain) not in bindings
                    ):
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
