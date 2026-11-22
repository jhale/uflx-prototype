"""Reference coordinates, physical charts and numerical integration."""

from functools import lru_cache
from itertools import product

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


def jacobian(chart, point, step):
    if not len(point):
        return np.empty((len(chart(point)), 0))
    return np.stack(
        [
            (chart(point + step * axis) - chart(point - step * axis)) / (2 * step)
            for axis in np.eye(len(point))
        ],
        axis=-1,
    )


def chart_of(piece, frame):
    def chart(point):
        return frame.context.at(
            {piece.reference_cell: point},
            lambda: np.real(np.asarray(frame.value(piece.parameterisation))),
        )

    return chart


def reference_point(chart, point, dim, step):
    """Invert a cell chart by Gauss-Newton iteration."""
    ref = np.full(dim, 1 / (dim + 1))
    if dim:
        for _ in range(30):
            correction = np.linalg.lstsq(
                jacobian(chart, ref, step), chart(ref) - np.real(point), rcond=None
            )[0]
            ref -= correction
            if np.linalg.norm(correction) < 1e-12:
                break
    return ref


def locate(domain, point, frame):
    for slot, piece in enumerate(domain):
        chart = chart_of(piece, frame)
        ref = reference_point(chart, point, domain.dimension, frame.context.spatial_step)
        cell = piece.reference_cell
        inside = (
            elements.inside_cell(cell, ref)
            if elements.is_basix_cell(cell)
            else np.all(ref > -1e-9) and np.sum(ref) < 1 + 1e-9
        )
        if inside and np.linalg.norm(chart(ref) - np.real(point)) < 1e-9:
            return slot
    raise ValueError(f"no cell of {domain} contains {point}")


def tangent_projector(domain, frame):
    points = frame.context.points
    step = frame.context.spatial_step
    if ("chart", domain) in points:
        tangent = jacobian(points[("chart", domain)], points[("ref", domain)], step)
    else:
        point = points[domain]
        slot = points.get(("piece", domain))
        if slot is None:
            slot = locate(domain, point, frame)
        chart = chart_of(domain[slot], frame)
        tangent = jacobian(chart, reference_point(chart, point, domain.dimension, step), step)
    return tangent @ np.linalg.pinv(tangent)


def side_reference(side, frame):
    source = side.source
    points = frame.context.points
    slot = points.get(("piece", source), 0)
    parent_slot, inclusion = side.entries[slot]
    point = points[source]
    if isinstance(source, Mesh):
        chart = chart_of(source[slot], frame)
        point = reference_point(chart, point, source.dimension, frame.context.spatial_step)
    target = frame.context.at(
        {inclusion.parameter_domain: point},
        lambda: np.real(np.asarray(frame.value(inclusion.parameterisation))),
    )
    return slot, parent_slot, inclusion, point, target


def integrate(domain, fn, frame):
    ctx = frame.context
    if isinstance(domain, AbstractCoordinateDomain):
        if elements.is_basix_cell(domain):
            rule = elements.cell_quadrature(domain)
        elif isinstance(domain, ReferenceSimplex) or domain == EuclideanSpace(0):
            rule = simplex_quadrature(domain.dimension)
        else:
            raise NotImplementedError(f"no reference quadrature for {domain}")
        return sum(weight * ctx.at({domain: np.asarray(point)}, fn) for point, weight in rule)

    def integrate_piece(slot, piece):
        chart = chart_of(piece, frame)

        def mapped():
            ref = ctx.points[piece.reference_cell]
            physical = chart(ref)
            J = jacobian(chart, ref, ctx.spatial_step)
            density = np.sqrt(abs(np.linalg.det(J.T @ J))) if len(ref) else 1.0
            return density * ctx.at(
                {
                    ("piece", domain): slot,
                    domain: physical,
                    domain.ambient_domain: physical,
                    ("ref", domain): ref,
                    ("chart", domain): chart,
                },
                fn,
            )

        return integrate(piece.reference_cell, mapped, frame)

    return sum(integrate_piece(slot, piece) for slot, piece in enumerate(domain))


def transformed(value, law, J):
    inverse = np.linalg.pinv(J)
    value = np.asarray(value)
    for axis, variance in enumerate(law.variances, start=value.ndim - len(law.variances)):
        matrix = inverse.T if variance == "covariant" else J
        value = np.moveaxis(np.tensordot(value, matrix, axes=([axis], [1])), -1, axis)
    scale = np.linalg.det(J) if J.shape[0] == J.shape[1] else np.sqrt(np.linalg.det(J.T @ J))
    return value / scale**law.weight
