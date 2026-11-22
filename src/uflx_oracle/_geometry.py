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
        return ad.linear(np.real, local.value(piece.parameterisation))

    return chart


def reference_point(chart, point, dim):
    """Invert a cell chart by Gauss-Newton iteration."""
    ref = np.full(dim, 1 / (dim + 1))
    if dim:
        for _ in range(30):
            residual = ad.subtract(chart(ref), ad.linear(np.real, point))
            correction = ad.matmul(ad.pinv(ad.jacobian(chart, ref)), residual)
            ref = ad.subtract(ref, correction)
            if converged(correction):
                break
    return ref


def converged(correction):
    # At a stationary primal point the inverse chart's tangents may still be changing.
    if isinstance(correction, ad.Dual):
        return converged(correction.value) and converged(correction.tangent)
    return np.linalg.norm(correction) < 1e-12


def locate(domain, point, frame):
    """Return the cell and its reference coordinate for a physical point."""
    slot = frame.cells.get(domain)
    if slot is not None:
        return slot, reference_point(chart_of(domain[slot], frame), point, domain.dimension)

    for slot, piece in enumerate(domain):
        chart = chart_of(piece, frame)
        ref = reference_point(chart, point, domain.dimension)
        cell = piece.reference_cell
        inside = (
            elements.inside_cell(cell, ad.primal(ref))
            if elements.is_basix_cell(cell)
            else np.all(ad.primal(ref) > -1e-9) and np.sum(ad.primal(ref)) < 1 + 1e-9
        )
        if inside and np.linalg.norm(ad.primal(chart(ref)) - np.real(ad.primal(point))) < 1e-9:
            return slot, ref
    raise ValueError(f"no cell of {domain} contains {point}")


def tangent_projector(domain, frame):
    slot, ref = locate(domain, frame.points[domain], frame)
    tangent = ad.jacobian(chart_of(domain[slot], frame), ref)
    return ad.matmul(tangent, ad.pinv(tangent))


def side_reference(side, frame):
    source = side.source
    slot = frame.cells.get(source, 0)
    parent_slot, inclusion = side.entries[slot]
    point = frame.points[source]
    if isinstance(source, Mesh):
        chart = chart_of(source[slot], frame)
        point = reference_point(chart, point, source.dimension)
    local = frame.at({inclusion.parameter_domain: point})
    target = ad.linear(np.real, local.value(inclusion.parameterisation))
    return slot, parent_slot, inclusion, point, target


def integrate(domain, fn, frame):
    if isinstance(domain, AbstractCoordinateDomain):
        if elements.is_basix_cell(domain):
            rule = elements.cell_quadrature(domain)
        elif isinstance(domain, ReferenceSimplex) or domain == EuclideanSpace(0):
            rule = simplex_quadrature(domain.dimension)
        else:
            raise NotImplementedError(f"no reference quadrature for {domain}")
        total = 0
        for point, weight in rule:
            total = ad.add(total, ad.multiply(weight, fn(frame.at({domain: np.asarray(point)}))))
        return total

    def integrate_piece(slot, piece):
        chart = chart_of(piece, frame)

        def mapped(local):
            ref = local.points[piece.reference_cell]
            physical = chart(ref)
            J = ad.jacobian(chart, ref)
            density = volume_factor(J) if len(ref) else 1.0
            sample = local.at(
                {domain: physical, domain.ambient_domain: physical}, cells={domain: slot}
            )
            return ad.multiply(density, fn(sample))

        return integrate(piece.reference_cell, mapped, frame)

    total = 0
    for slot, piece in enumerate(domain):
        total = ad.add(total, integrate_piece(slot, piece))
    return total


def volume_factor(J):
    gram = ad.matmul(ad.linear(np.transpose, J), J)
    return ad.sqrt(ad.absolute(ad.determinant(gram)))


def transformed(value, law, J):
    inverse = ad.pinv(J)
    value = ad.asarray(value)
    for axis, variance in enumerate(
        law.variances, start=np.ndim(ad.primal(value)) - len(law.variances)
    ):
        matrix = ad.linear(np.transpose, inverse) if variance == "covariant" else J
        value = ad.bilinear(
            lambda v, m: np.moveaxis(np.tensordot(v, m, axes=([axis], [1])), -1, axis),
            value,
            matrix,
        )
    m, n = np.shape(ad.primal(J))
    scale = ad.determinant(J) if m == n else volume_factor(J)
    return ad.divide(value, ad.power(scale, law.weight))
