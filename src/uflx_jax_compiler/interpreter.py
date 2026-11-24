# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Evaluate UFLx expressions and trace finite element forms to local tensors."""

import jax
import jax.numpy as jnp

from uflx.algorithms import memoized, operands
from uflx.nodes import (
    AbstractDomain,
    AbstractExpression,
    AbstractExtent,
    AbstractTopology,
    Integral,
    Mesh,
)

from ._basis_tables import BasisTable, basis_tables
from ._context import Sample
from ._external import Pointwise as Pointwise
from ._finite_elements import local_basis
from ._reference import reference_kernel
from ._semantics import evaluate


def _tabulate_slot(action, position, dim, basis):
    def tabulated(*values):
        def entry(i):
            return action(*values[:position], basis(i), *values[position:])

        return jax.vmap(entry)(jnp.arange(dim))

    return tabulated


def _check_quadrature(expression, inputs):
    declared_tables = {key for key in inputs if isinstance(key, BasisTable)}
    tabulated = {table.point_source for table in declared_tables}

    @memoized
    def visit(walk, node):
        if (
            isinstance(node, Integral)
            and node not in inputs
            and (node.integration_domain.dimension or node in tabulated)
        ):
            domain = node.integration_domain
            cells = domain.topology.cell_types if isinstance(domain, Mesh) else (domain,)
            if any((node, cell) not in inputs for cell in cells):
                raise ValueError("supply an explicit quadrature rule for each integral", node)
        # Derived index labels can name the integral that quadrature already expanded.
        fields = [name for name in node.argument_names if name not in node.labels]
        for child in operands(node, fields):
            walk(child)

    visit(expression)
    if declared_tables and not declared_tables.issubset(basis_tables(expression)):
        raise ValueError("basis table is not requested by this expression")


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

    With apply_quadrature, declare the runtime point and weight arrays as inputs.
    Their shapes determine any shared symbolic extent. An unexpanded reference
    integral can also take its (points, weights) pair as an input.
    For a mesh with different cell types, declare (integral, reference_cell)
    inputs for the respective rules. A point integral is evaluated exactly.

    For explicit FE tables, pass selected requests from basis_tables(expression)
    as inputs. For each request, supply request.element.tabulate(request.order, points).
    Close over a fixed rule or precomputed table in a Python function before JIT.

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
    kernel = reference_kernel(expression, inputs, _kernel) or _kernel(expression, inputs)
    kernel.inputs = inputs
    return kernel


def _kernel(expression, inputs):
    # Runtime extents are structural types. Bind them from array shapes while
    # tracing, so array lengths do not become additional scalar JIT arguments.
    shaped_inputs = [
        (position, key)
        for position, key in enumerate(inputs)
        if isinstance(key, AbstractExpression)
        and key.domain is None
        and any(isinstance(extent, AbstractExtent) for extent in key.value_shape)
        and not key.argument_spaces
        and not key.free_indices
        and not isinstance(key, Integral)
    ]

    def action(*values):
        environment = {}
        points = {}
        for key, value in zip(inputs, values[: len(inputs)], strict=True):
            if isinstance(key, (AbstractDomain, AbstractTopology)):
                points[key] = jnp.asarray(value, dtype=float)
            else:
                environment[key] = value
        inferred = {}
        for position, key in shaped_inputs:
            data = values[position]
            if callable(data) or isinstance(data, (AbstractExpression, dict)):
                continue
            shape = jnp.asarray(data).shape
            if len(shape) != len(key.value_shape):
                raise ValueError(f"expected array shape {key.value_shape}, got {shape}")
            for extent, length in zip(key.value_shape, shape, strict=True):
                expected = (
                    inferred.setdefault(extent, length)
                    if isinstance(extent, AbstractExtent)
                    else extent
                )
                if length != expected:
                    raise ValueError(
                        f"inconsistent array extent: expected {expected}, got {length}"
                    )
        for extent, length in inferred.items():
            if (
                extent in environment
                and isinstance(environment[extent], int)
                and environment[extent] != length
            ):
                raise ValueError(
                    f"inconsistent array extent: expected {length}, got {environment[extent]}"
                )
            environment.setdefault(extent, length)
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
