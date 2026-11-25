# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Numerically integrate test graphs with a caller-selected polynomial degree."""

import basix

from uflx.algorithms import expression_operands, memoized
from uflx.nodes import Integral, Mesh
from uflx_basix.nodes import BasixCell
from uflx_jax_compiler import reference_trace


def integrate_at(expression, bindings, *arguments, degree):
    """Give each integral in a reference check a rule for its own reference cell."""
    inputs = dict(bindings)

    @memoized
    def visit(walk, node):
        if isinstance(node, Integral) and node.integration_domain.dimension:
            domain = node.integration_domain
            cells = domain.topology.cell_types if isinstance(domain, Mesh) else (domain,)
            for cell in set(cells):
                if isinstance(cell, BasixCell):
                    rule = basix.make_quadrature(cell.cell_type, degree)
                else:
                    rule = cell.quadrature(degree)
                inputs[node, cell] = rule
        for child in expression_operands(node):
            walk(child)

    visit(expression)
    return reference_trace(expression, *inputs)(*inputs.values(), *arguments)
