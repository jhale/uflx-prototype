# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Extract pointwise geometry from a reference-cell integral."""

from uflx import nodes
from uflx._src.algorithms.traversal import resolve
from uflx.algorithms import bottom_up, expression_operands, map_values
from uflx.calculus import integral
from uflx_extensions.runtime_arrays import runtime_array

from ._basis_tables import keeps_quadrature_point


def extract_geometry(reference_integral, cell_map):
    """Return a reference integral and a mapping from pointwise operands to geometry expressions.

    Extract maximal real expressions built from the cell parameterisation,
    reference coordinates and literals, with no free indices. Each operand has
    the shape of its value at one reference point. The same geometry expression
    yields the same operand across integrals and quadrature rules.
    Leave mixed spatial derivatives and operations that resample fields intact.
    """
    if not isinstance(reference_integral, nodes.Integral):
        raise TypeError("expected a single reference-cell Integral")
    cell = cell_map.reference_cell
    if reference_integral.integration_domain != cell:
        raise ValueError("the integral must be on the geometry map's reference cell")
    integrand = reference_integral.expression
    chart = cell_map.parameterisation

    # False means fixed, True means coordinate-dependent, None needs other data.
    dependencies = {}

    def classify(node):
        depends = None
        if node is chart:
            depends = True
        elif isinstance(
            node, (nodes.Integer, nodes.Float, nodes.Complex, nodes.Boolean, nodes.Zero)
        ):
            depends = False
        elif isinstance(node, nodes.Coordinate) and node.domain == cell:
            depends = False
        elif keeps_quadrature_point(node) and not isinstance(node, nodes.PullBack):
            children = [dependencies[child] for child in expression_operands(node)]
            if None not in children and node.domain in (None, cell):
                depends = any(children)
        if isinstance(node, nodes.Indexed) and any(
            isinstance(i, nodes.GatheredIndex) for i in node.indices
        ):
            depends = None
        if node.argument_spaces or any(
            isinstance(n, nodes.AbstractExtent)
            for n in (*node.value_shape, *(n for _, n in node.free_indices))
        ):
            depends = None
        dependencies[node] = depends
        return node

    bottom_up(integrand, classify)
    geometry = {}
    rewritten = {}

    def split(node):
        if dependencies[node] and not node.free_indices and node.scalar_type == "real":
            operand = runtime_array(f"geometry_{node.digest:032x}", node.value_shape)
            geometry[operand] = node
            return operand
        # A point value cannot supply a spatial derivative or a value at a different point.
        if not keeps_quadrature_point(node) or isinstance(
            node, (nodes.Gradient, nodes.Divergence, nodes.Curl, nodes.PullBack, nodes.Conditional)
        ):
            return node
        yield from expression_operands(node)
        return map_values(node, rewritten.__getitem__)

    residual = resolve(integrand, split, rewritten)
    return integral(residual, cell), geometry
