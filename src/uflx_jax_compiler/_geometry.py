# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Reference coordinates, physical charts and numerical integration."""

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractCoordinateDomain, Mesh, Product

from . import _tensors
from ._basis_tables import quadrature_tables
from ._context import PointTabulation


def chart_of(piece, site, value):
    def chart(point):
        at = site.at({piece.reference_cell: point})
        return jnp.real(value(piece.parameterisation, at=at))

    return chart


def side_reference(side, site, value):
    source = side.source
    slot = site.slot(source)
    parent_slot, inclusion = side.entries[slot]
    point = site.points[source]
    if isinstance(source, Mesh):
        reference = site.require_reference("Physical side-map inversion")
        chart = chart_of(source[slot], site, value)
        point = reference.reference_point(chart, point, source.dimension)
    at = site.at({inclusion.parameter_domain: point})
    target = jnp.real(value(inclusion.parameterisation, at=at))
    return slot, parent_slot, inclusion, point, target


def quadrature_sum(
    result, cell, point_source, rule, integrand, product, site, bindings, value, *, eager=False
):
    """Evaluate reference samples and reduce their leading quadrature axis."""
    points, weights = rule
    points, weights = jnp.asarray(points, dtype=float), jnp.asarray(weights)
    if weights.ndim != 1 or points.shape != (weights.size, cell.dimension):
        raise ValueError("quadrature needs one reference point per weight", result)
    tables = quadrature_tables(point_source, cell, bindings, weights.size)
    if not weights.size:
        return jnp.zeros(_tensors.shape(result, bindings), dtype=_tensors.dtype(result))

    factorized = (
        bool(tables)
        and isinstance(product, Product)
        and not (product.value_shape or product.free_indices or product.argument_spaces)
    )

    def sample(row, point, *rows):
        at_point = (
            {cell: PointTabulation(point, dict(zip(tables, rows, strict=True)))} if tables else None
        )
        at = site.at({cell: point}, tabulations=at_point)
        if factorized:
            return value(product.first, at=at), value(product.second, at=at)
        return integrand(at, row)

    if eager:
        samples = jnp.stack(
            [sample(i, p, *(rows[i] for rows in tables.values())) for i, p in enumerate(points)]
        )
    else:
        samples = jax.vmap(sample)(jnp.arange(weights.size), points, *tables.values())
    if factorized:
        # vmap over independent test/trial slots turns this dot into a
        # matrix product without forming a (test, trial, point) array.
        left, right = samples
        return jnp.dot(left, weights * right)
    return jnp.tensordot(weights, samples, axes=1)


def integrate(integral, bindings, fn, site, value):
    domain = integral.integration_domain

    def on_reference_cell(cell, integrand):
        rule = bindings.get((integral, cell), bindings.get(integral))
        if rule is None:
            if cell.dimension == 0:
                return integrand(site.at({cell: jnp.empty(0)}))
            raise ValueError("supply an explicit quadrature rule for each integral", integral)
        return quadrature_sum(
            integral,
            cell,
            integral,
            rule,
            lambda at, row: integrand(at),
            integral.expression if cell == domain else None,
            site,
            bindings,
            value,
        )

    if isinstance(domain, AbstractCoordinateDomain):
        return on_reference_cell(domain, fn)

    reference = site.require_reference("Physical mesh integration")
    return reference.integrate_mesh(domain, fn, site, value, on_reference_cell)
