# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Evaluate UFLx expressions and trace finite element forms to local tensors."""

import jax
import jax.numpy as jnp

from uflx.algorithms import memoized, operands
from uflx.nodes import AbstractDomain, AbstractExpression, AbstractReferenceCell, AbstractTopology, Integral, Mesh

from ._context import Sample
from ._external import Pointwise as Pointwise
from ._finite_elements import local_basis, tabulation_space
from ._reference import reference_kernel
from ._semantics import evaluate


def _tabulate_slot(action, position, dim, basis):
    def tabulated(*values):
        def entry(i):
            return action(*values[:position], basis(i), *values[position:])

        return jax.vmap(entry)(jnp.arange(dim))

    return tabulated


def _check_quadrature(expression, inputs):
    tabulated = {
        key[0] for key in inputs
        if isinstance(key, tuple) and len(key) == 2 and isinstance(key[0], Integral)
        and tabulation_space(key[1]) is not None
    }
    if any(not isinstance(I.integration_domain, AbstractReferenceCell) for I in tabulated):
        raise ValueError("declare runtime tabulations on lowered reference-cell integrals")

    @memoized
    def visit(walk, node):
        if isinstance(node, Integral) and (node.integration_domain.dimension or node in tabulated) and node not in inputs:
            domain = node.integration_domain
            cells = domain.topology.cell_types if isinstance(domain, Mesh) else (domain,)
            if any((node, cell) not in inputs for cell in cells):
                raise ValueError("supply an explicit quadrature rule for each integral", node)
        tabulated.discard(node)
        # Derived index labels can name the integral that quadrature already expanded.
        fields = [name for name in node.argument_names if name not in node.labels]
        for child in operands(node, fields):
            walk(child)

    visit(expression)
    if tabulated:
        raise ValueError("a runtime tabulation must refer to an integral in the expression")


def trace(expression: AbstractExpression, *inputs):
    """Return a JAX-traceable function for an expression or form.

    Each Basix form slot contributes a leading tensor axis in declaration order.
    The axis follows the space's packed local DOF order, including all cell slots.
    A product of Basix spaces concatenates its members' DOFs on one axis.
    Use uflx.forms.apply or partial to take an action on a specific field.

    Call the kernel with one value per declared input, followed by one value for
    each form slot without a Basix basis. A field input takes its coefficients,
    a domain input takes an evaluation point, and a free index takes its component.
    An index table input takes an integer array with the declared table shape.

    Apply quadrature to each reference integral before tracing, or declare the
    integral as an input and pass its (points, weights) on the reference cell.
    For a mesh with different cell types, declare (integral, reference_cell)
    inputs for the respective rules. A point integral is evaluated exactly.

    For a reference-cell integral I with runtime quadrature, declare (I, V)
    as a further input to supply the Basix basis of space V at those points.
    Pass element.tabulate(order, points), with axes (derivative, quadrature
    point, local element DOF, value component). Supply at least the spatial
    derivative order used by the integrand. Fields and form slots in V share
    that table. Use (I, BasixBasis(V, slot)) to select an individual cell slot.

    For separable real forms, fixed quadrature sums over reference basis functions
    are precomputed and contracted with the cell data at evaluation.

    A blocked Basix field accepts flat DOFs or one block per scalar basis DOF.
    Unselected free indices become leading array axes in free-index order.
    An external operation takes a JAX-traceable implementation. Field operands
    are callable. Wrap the implementation in Pointwise to receive their values
    at the current point. Bind fixed inputs with functools.partial. To sample a
    particular cell, pull the expression back to its reference cell before tracing.
    """
    keys = set(inputs)
    if len(keys) != len(inputs):
        raise ValueError("each input must be declared once")
    _check_quadrature(expression, keys)

    return reference_kernel(expression, inputs, _kernel) or _kernel(expression, inputs)


def _kernel(expression, inputs):
    def action(*values):
        environment = {}
        points = {}
        for key, value in zip(inputs, values[: len(inputs)], strict=True):
            if isinstance(key, (AbstractDomain, AbstractTopology)):
                points[key] = jnp.asarray(value, dtype=float)
            else:
                environment[key] = value
        sample = Sample(points, {})
        return jnp.asarray(evaluate(expression, environment, sample, values[len(inputs) :]))

    arity = len(inputs) + len(expression.argument_spaces)
    # Wrap the last slot first so the first declared basis is the outermost axis.
    for i, space in reversed(tuple(enumerate(expression.argument_spaces))):
        basis = local_basis(space)
        if basis is not None:
            action = _tabulate_slot(action, len(inputs) + i, *basis)
            arity -= 1

    def kernel(*values):
        if len(values) != arity:
            raise TypeError(f"expected {arity} positional values, got {len(values)}")
        return action(*values)

    return kernel
