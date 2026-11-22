"""Reference coordinates, physical charts and numerical integration."""

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractCoordinateDomain, Mesh

from . import _finite_elements as elements
from . import _tensors


def chart_of(piece, frame):
    def chart(point):
        local = frame.at({piece.reference_cell: point})
        return jnp.real(local.value(piece.parameterisation))

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
    point = frame.points[domain]

    def candidate(piece):
        chart = chart_of(piece, frame)
        ref = reference_point(chart, point, domain.dimension)
        Q, _ = jnp.linalg.qr(jax.jacfwd(chart)(ref))
        return ref, chart(ref), Q @ Q.T

    if domain in frame.cells:
        return candidate(domain[frame.cells[domain]])[2]
    projection = jnp.full((len(point), len(point)), jnp.nan)
    found = False
    for piece in domain:
        ref, physical, matrix = candidate(piece)
        cell = piece.reference_cell
        inside = (
            elements.inside_cell(cell, ref)
            if elements.is_basix_cell(cell)
            else jnp.all(ref > -1e-9) & (jnp.sum(ref) < 1 + 1e-9)
        )
        valid = inside & (jnp.linalg.norm(physical - jnp.real(point)) < 1e-9)
        projection = jnp.where(valid & ~jnp.asarray(found), matrix, projection)
        found = found | valid
    if not isinstance(found, jax.core.Tracer) and not found:
        raise ValueError(f"no cell of {domain} contains {point}")
    return projection


def side_reference(side, frame):
    source = side.source
    slot = frame.cells.get(source, 0)
    parent_slot, inclusion = side.entries[slot]
    point = frame.points[source]
    if isinstance(source, Mesh):
        chart = chart_of(source[slot], frame)
        point = reference_point(chart, point, source.dimension)
    local = frame.at({inclusion.parameter_domain: point})
    target = jnp.real(local.value(inclusion.parameterisation))
    return slot, parent_slot, inclusion, point, target


def integrate(domain, fn, frame):
    if isinstance(domain, AbstractCoordinateDomain):
        points, weights = elements.cell_quadrature(domain)
        samples = jax.vmap(lambda p: fn(frame.at({domain: p})))(jnp.asarray(points))
        return jnp.tensordot(jnp.asarray(weights), samples, axes=1)

    def integrate_piece(slot, piece):
        chart = chart_of(piece, frame)

        def mapped(local):
            ref = local.points[piece.reference_cell]
            physical = chart(ref)
            J = jax.jacfwd(chart)(ref)
            density = jnp.abs(jacobian_scale(J))
            sample = local.at(
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
