from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import Absolute, Gradient, Integer, Inverse, SquareRoot
from .tensors import determinant, matrix_product, transpose

if TYPE_CHECKING:
    from .nodes.domains import AbstractCoordinateDomain
    from .nodes.expressions import AbstractExpression
    from .nodes.maps import DomainMap
    from .nodes.meshes import CellMap


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_geometric_map", "_metric", "density", "inverse_jacobian", "jacobian"]


def jacobian(D: DomainMap | CellMap) -> AbstractExpression:
    source, _ = _geometric_map(D)
    return Gradient(D.parameterisation, source)


def _metric(D: DomainMap | CellMap) -> AbstractExpression:
    # JᵀJ, the metric that the parameterisation induces on its parameter domain.
    J = jacobian(D)
    return matrix_product(transpose(J), J)


def density(D: DomainMap | CellMap) -> AbstractExpression:
    # sqrt(det G), with the closed form |det J| for a cell of the ambient dimension.
    source, target = _geometric_map(D)
    k, n = source.dimension, target.dimension
    if k == 0:
        return Integer(1)
    if k == n:
        return Absolute(determinant(jacobian(D)))
    return SquareRoot(determinant(_metric(D)))


def inverse_jacobian(D: DomainMap | CellMap) -> AbstractExpression:
    # The pseudo-inverse of J, which is the inverse for a cell of the ambient dimension.
    return Inverse(jacobian(D))


def _geometric_map(
    D: DomainMap | CellMap,
) -> tuple[AbstractCoordinateDomain, AbstractCoordinateDomain]:
    from .nodes import AbstractCoordinateDomain, Mesh

    source, target = D.parameter_domain, D.ambient_domain
    if not isinstance(source, AbstractCoordinateDomain):
        raise TypeError("geometry requires a coordinate parameter domain")
    if isinstance(target, Mesh):
        return source, target.ambient_coordinate_domain
    if isinstance(target, AbstractCoordinateDomain):
        return source, target
    raise TypeError("geometry requires a coordinate map or mesh cell map")
