"""Reference coordinates, physical charts and numerical integration."""

from functools import lru_cache
from itertools import product

import numpy as np

from uflx.nodes import AbstractCoordinateDomain, EuclideanSpace, Mesh
from uflx_basix.nodes import ReferenceSimplex

from . import _dual as ad
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
        return np.real(ad.asarray(local.value(piece.parameterisation)))

    return chart


def reference_point(chart, point, dim):
    """Invert a cell chart by Gauss-Newton iteration."""
    ref = np.full(dim, 1 / (dim + 1))
    if dim:
        for _ in range(30):
            correction = np.linalg.pinv(ad.jacobian(chart, ref)) @ (chart(ref) - np.real(point))
            ref = ref - correction
            if converged(correction):
                break
    return ref


def converged(correction):
    # At a stationary primal point the inverse chart's tangents may still be changing.
    if isinstance(correction, ad.Dual):
        return converged(correction.value) and converged(correction.tangent)
    return np.linalg.norm(correction) < 1e-12


def locate(domain, point, frame):
    for slot, piece in enumerate(domain):
        chart = chart_of(piece, frame)
        ref = reference_point(chart, point, domain.dimension)
        cell = piece.reference_cell
        inside = (
            elements.inside_cell(cell, ad.primal(ref))
            if elements.is_basix_cell(cell)
            else np.all(ref > -1e-9) and np.sum(ref) < 1 + 1e-9
        )
        if inside and np.linalg.norm(ad.primal(chart(ref) - np.real(point))) < 1e-9:
            return slot
    raise ValueError(f"no cell of {domain} contains {point}")


def tangent_projector(domain, frame):
    point = frame.points[domain]
    slot = frame.points.get(("piece", domain))
    if slot is None:
        slot = locate(domain, point, frame)
    chart = chart_of(domain[slot], frame)
    tangent = ad.jacobian(chart, reference_point(chart, point, domain.dimension))
    return tangent @ np.linalg.pinv(tangent)


def side_reference(side, frame):
    source = side.source
    points = frame.points
    slot = points.get(("piece", source), 0)
    parent_slot, inclusion = side.entries[slot]
    point = points[source]
    if isinstance(source, Mesh):
        chart = chart_of(source[slot], frame)
        point = reference_point(chart, point, source.dimension)
    local = frame.at({inclusion.parameter_domain: point})
    target = np.real(ad.asarray(local.value(inclusion.parameterisation)))
    return slot, parent_slot, inclusion, point, target


def integrate(domain, fn, frame):
    if isinstance(domain, AbstractCoordinateDomain):
        if elements.is_basix_cell(domain):
            rule = elements.cell_quadrature(domain)
        elif isinstance(domain, ReferenceSimplex) or domain == EuclideanSpace(0):
            rule = simplex_quadrature(domain.dimension)
        else:
            raise NotImplementedError(f"no reference quadrature for {domain}")
        return sum(weight * fn(frame.at({domain: np.asarray(point)})) for point, weight in rule)

    def integrate_piece(slot, piece):
        chart = chart_of(piece, frame)

        def mapped(local):
            ref = local.points[piece.reference_cell]
            physical = chart(ref)
            J = ad.jacobian(chart, ref)
            density = np.sqrt(abs(np.linalg.det(J.T @ J))) if len(ref) else 1.0
            return density * fn(
                local.at(
                    {("piece", domain): slot, domain: physical, domain.ambient_domain: physical}
                )
            )

        return integrate(piece.reference_cell, mapped, frame)

    return sum(integrate_piece(slot, piece) for slot, piece in enumerate(domain))


def transformed(value, law, J):
    inverse = np.linalg.pinv(J)
    value = ad.asarray(value)
    for axis, variance in enumerate(law.variances, start=value.ndim - len(law.variances)):
        matrix = inverse.T if variance == "covariant" else J
        value = np.moveaxis(np.tensordot(value, matrix, axes=([axis], [1])), -1, axis)
    scale = np.linalg.det(J) if J.shape[0] == J.shape[1] else np.sqrt(np.linalg.det(J.T @ J))
    return value / scale**law.weight
