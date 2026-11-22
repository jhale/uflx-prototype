"""Reference coordinates, physical charts and numerical integration."""

from functools import lru_cache
from itertools import product

import jax
import jax.numpy as jnp
import numpy as np

from uflx.nodes import AbstractCoordinateDomain, EuclideanSpace, Mesh
from uflx_basix.nodes import ReferenceSimplex

from . import _finite_elements as elements


@lru_cache
def simplex_quadrature(dimension):
    """Map a tensor product of Gauss rules on [0, 1] onto a unit simplex."""
    nodes, weights = np.polynomial.legendre.leggauss(6 if dimension == 1 else 5)
    interval = tuple(zip((nodes + 1) / 2, weights / 2, strict=True))
    rule = []
    for factors in product(interval, repeat=dimension):
        point, remaining, weight = [], 1.0, 1.0
        for node, w in factors:
            point.append(remaining * node)
            weight *= remaining * w
            remaining *= 1 - node
        rule.append((tuple(point), weight))
    return tuple(rule)


def chart_of(piece, frame):
    def chart(point):
        local = frame.at({piece.reference_cell: point})
        return jnp.real(local.value(piece.parameterisation))

    return chart


def reference_point(chart, point, dim):
    """Solve for the reference point and differentiate its stationarity condition."""
    initial = jnp.full(dim, 1 / (dim + 1))
    if not dim:
        return initial
    jacobian = jax.jacfwd(chart)

    def stationarity(ref):
        return jacobian(ref).T @ (chart(ref) - jnp.real(point))

    def solve(_, ref):
        def step(state):
            i, ref, _ = state
            correction = jnp.linalg.pinv(jacobian(ref)) @ (chart(ref) - jnp.real(point))
            return i + 1, ref - correction, jnp.linalg.norm(correction)

        return jax.lax.while_loop(
            lambda state: (state[0] < 30) & (state[2] >= 1e-12),
            step,
            (0, ref, jnp.inf),
        )[1]

    def tangent_solve(linear, rhs):
        return jnp.linalg.solve(jax.jacfwd(linear)(jnp.zeros_like(rhs)), rhs)

    return jax.lax.custom_root(stationarity, initial, solve, tangent_solve)


def tangent_projector(domain, frame):
    point = frame.points[domain]

    def candidate(piece):
        chart = chart_of(piece, frame)
        ref = reference_point(chart, point, domain.dimension)
        tangent = jax.jacfwd(chart)(ref)
        return ref, chart(ref), tangent @ jnp.linalg.pinv(tangent)

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
        if elements.is_basix_cell(domain):
            rule = elements.cell_quadrature(domain)
        elif isinstance(domain, ReferenceSimplex) or domain == EuclideanSpace(0):
            rule = simplex_quadrature(domain.dimension)
        else:
            raise NotImplementedError(f"no reference quadrature for {domain}")
        points, weights = zip(*rule, strict=True)
        samples = jax.vmap(lambda p: fn(frame.at({domain: p})))(jnp.asarray(points))
        return jnp.tensordot(jnp.asarray(weights), samples, axes=1)

    def integrate_piece(slot, piece):
        chart = chart_of(piece, frame)

        def mapped(local):
            ref = local.points[piece.reference_cell]
            physical = chart(ref)
            J = jax.jacfwd(chart)(ref)
            density = volume_factor(J) if len(ref) else 1.0
            sample = local.at(
                {domain: physical, domain.ambient_domain: physical}, cells={domain: slot}
            )
            return density * fn(sample)

        return integrate(piece.reference_cell, mapped, frame)

    total = 0
    for slot, piece in enumerate(domain):
        total = total + integrate_piece(slot, piece)
    return total


def volume_factor(J):
    return jnp.sqrt(jnp.abs(jnp.linalg.det(J.T @ J)))


def transformed(value, law, J):
    value = jnp.asarray(value)
    for axis, variance in enumerate(law.variances, start=value.ndim - len(law.variances)):
        matrix = jnp.linalg.pinv(J).T if variance == "covariant" else J
        value = jnp.moveaxis(jnp.tensordot(value, matrix, axes=([axis], [1])), -1, axis)
    scale = jnp.linalg.det(J) if J.shape[0] == J.shape[1] else volume_factor(J)
    return value / scale**law.weight
