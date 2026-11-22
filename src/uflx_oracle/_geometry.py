"""Reference coordinates, physical charts and numerical integration."""

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractCoordinateDomain, Mesh

from . import _finite_elements as elements
from . import _tensors


def chart_of(piece, frame):
    def chart(point):
        site = frame.site.at({piece.reference_cell: point})
        return jnp.real(frame.value(piece.parameterisation, at=site))

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


def tangent_projector(domain, frame):
    chart = chart_of(domain[frame.site.slot(domain)], frame)
    ref = reference_point(chart, frame.site.points[domain], domain.dimension)
    Q, _ = jnp.linalg.qr(jax.jacfwd(chart)(ref))
    return Q @ Q.T


def side_reference(side, frame):
    source = side.source
    slot = frame.site.slot(source)
    parent_slot, inclusion = side.entries[slot]
    point = frame.site.points[source]
    if isinstance(source, Mesh):
        chart = chart_of(source[slot], frame)
        point = reference_point(chart, point, source.dimension)
    site = frame.site.at({inclusion.parameter_domain: point})
    target = jnp.real(frame.value(inclusion.parameterisation, at=site))
    return slot, parent_slot, inclusion, point, target


def integrate(domain, fn, frame):
    if isinstance(domain, AbstractCoordinateDomain):
        points, weights = elements.cell_quadrature(domain)
        samples = jax.vmap(lambda p: fn(frame.site.at({domain: p})))(jnp.asarray(points))
        return jnp.tensordot(jnp.asarray(weights), samples, axes=1)

    def integrate_piece(slot, piece):
        chart = chart_of(piece, frame)

        def mapped(site):
            ref = site.points[piece.reference_cell]
            physical = chart(ref)
            J = jax.jacfwd(chart)(ref)
            density = jnp.abs(jacobian_scale(J))
            sample = site.at(
                {domain: physical, domain.ambient_domain: physical}, cells={domain: slot}
            )
            return density * fn(sample)

        return integrate(piece.reference_cell, mapped, frame)

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
