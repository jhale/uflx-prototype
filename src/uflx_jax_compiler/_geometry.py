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


def reference_point(chart, point, dim):
    """Invert a cell chart by Newton iteration (Gauss-Newton for an embedded cell)."""
    initial = jnp.full(dim, 1 / (dim + 1))
    if not dim:
        return initial
    jacobian = jax.jacfwd(chart)

    def correction(ref):
        return _tensors.solve(jacobian(ref), chart(ref) - jnp.real(point))

    def iterate(correction, ref):
        def step(state):
            i, ref, _ = state
            delta = correction(ref)
            return i + 1, ref - delta, jnp.linalg.norm(delta)

        return jax.lax.while_loop(
            lambda state: (state[0] < 30) & (state[2] >= 1e-12),
            step,
            (0, ref, jnp.inf),
        )[1]

    def tangent_solve(linear, rhs):
        return jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs)

    return jax.lax.custom_root(correction, initial, iterate, tangent_solve)


def tangent_projector(domain, site, value):
    chart = chart_of(domain[site.slot(domain)], site, value)
    ref = reference_point(chart, site.points[domain], domain.dimension)
    Q, _ = jnp.linalg.qr(jax.jacfwd(chart)(ref))
    return Q @ Q.T


def side_reference(side, site, value):
    source = side.source
    slot = site.slot(source)
    parent_slot, inclusion = side.entries[slot]
    point = site.points[source]
    if isinstance(source, Mesh):
        chart = chart_of(source[slot], site, value)
        point = reference_point(chart, point, source.dimension)
    at = site.at({inclusion.parameter_domain: point})
    target = jnp.real(value(inclusion.parameterisation, at=at))
    return slot, parent_slot, inclusion, point, target


def quadrature_sum(result, cell, point_source, rule, integrand, product, site, bindings, value):
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

    def sample(point, *rows):
        at_point = (
            {cell: PointTabulation(point, dict(zip(tables, rows, strict=True)))} if tables else None
        )
        at = site.at({cell: point}, tabulations=at_point)
        if factorized:
            return value(product.first, at=at), value(product.second, at=at)
        return integrand(at)

    samples = jax.vmap(sample)(points, *tables.values())
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
            integrand,
            integral.expression if cell == domain else None,
            site,
            bindings,
            value,
        )

    if isinstance(domain, AbstractCoordinateDomain):
        return on_reference_cell(domain, fn)

    def integrate_piece(slot, piece):
        chart = chart_of(piece, site, value)

        def mapped(site):
            ref = site.points[piece.reference_cell]
            physical = chart(ref)
            J = jax.jacfwd(chart)(ref)
            density = jnp.abs(jacobian_scale(J))
            sample = site.at(
                {domain: physical, domain.ambient_domain: physical}, cells={domain: slot}
            )
            return density * fn(sample)

        return on_reference_cell(piece.reference_cell, mapped)

    total = 0
    for slot, piece in enumerate(domain):
        total = total + integrate_piece(slot, piece)
    return total


def jacobian_scale(J):
    """Return the signed determinant for a square chart or the density for an embedded chart."""
    if J.shape[0] == J.shape[1]:
        return _tensors.determinant(J)
    _, R = jnp.linalg.qr(J)
    return jnp.abs(jnp.prod(jnp.diag(R)))


def transformed(value, law, J):
    value = jnp.asarray(value)
    for axis, variance in enumerate(law.variances, start=value.ndim - len(law.variances)):
        matrix = _tensors.inverse(J).T if variance == "covariant" else J
        value = jnp.moveaxis(jnp.tensordot(value, matrix, axes=([axis], [1])), -1, axis)
    if law.weight:
        value = value / jacobian_scale(J) ** law.weight
    return value
