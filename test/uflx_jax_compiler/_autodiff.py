# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""JAX differentiation rules for numerical reference evaluation of unlowered forms."""

import jax
import jax.numpy as jnp

from uflx.nodes import Mesh
from uflx_jax_compiler import _tensors
from uflx_jax_compiler._context import BoundExpression
from uflx_jax_compiler._geometry import chart_of, side_reference


def directional(node, arguments, read):
    def perturbed(epsilon):
        def seed(variable, point, direction):
            def varied(site, slots):
                return read(point, slots, at=site) + epsilon * read(direction, slots, at=site)

            return BoundExpression(variable, varied)

        overrides = {
            variable: seed(variable, p, d)
            for variable, p, d in zip(
                jax.tree.leaves(node.variable),
                jax.tree.leaves(node.point),
                jax.tree.leaves(node.direction),
                strict=True,
            )
        }
        value = read(node.expression, arguments, bind=overrides)
        # An integer literal denotes a real constant in a differentiable expression.
        return jnp.asarray(value, dtype=jnp.result_type(value, 0.0))

    return jax.jvp(perturbed, (0.0,), (1.0,))[1]


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


def gradient(x, domain, arguments, site, scoped_value, point):
    def at_point(p):
        result = scoped_value(x, arguments, at=site.differentiating(domain, p))
        return jnp.asarray(result, dtype=jnp.result_type(result, 0.0))

    g = jax.jacfwd(at_point)(point)
    if isinstance(domain, Mesh) and domain.dimension < point.size:
        g = g @ tangent_projector(domain, site, scoped_value)
    return g


def push_forward(field, mesh, law, site, value):
    slot = site.slot(mesh)
    chart = chart_of(mesh[slot], site, value)
    ref = reference_point(chart, site.points[mesh], mesh.dimension)
    topology = mesh.topology
    site = site.at({topology: ref}, cells={topology: slot})
    return transformed(field(site), law, jax.jacfwd(chart)(ref))


def conormal(side, conormals, site, value):
    slot, parent_slot, inclusion, ref, target = side_reference(side, site, value)
    J = jax.jacfwd(chart_of(side.parent[parent_slot], site, value))(target)
    site = site.at({inclusion.parameter_domain: ref})
    normal = jnp.real(value(conormals[slot], at=site))
    vector = _tensors.inverse(J).T @ normal
    return vector / jnp.linalg.norm(vector)


def integrate_mesh(domain, fn, site, value, on_reference_cell):
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
