from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import Gradient, SquareRoot
from .tensors import determinant, matrix_product, transpose

if TYPE_CHECKING:
    from .nodes.domains import AbstractCoordinateDomain
    from .nodes.expressions import AbstractExpression
    from .nodes.maps import DomainMap
    from .nodes.meshes import CellMap


def jacobian(map: DomainMap | CellMap) -> AbstractExpression:
    source, _ = geometric_map(map)
    return Gradient(map.parameterisation, source)


def _metric(D: DomainMap | CellMap) -> AbstractExpression:
    # JᵀJ, the metric that the parameterisation induces on its parameter domain.
    J = jacobian(D)
    return matrix_product(transpose(J), J)


def density(map: DomainMap | CellMap) -> AbstractExpression:
    return SquareRoot(determinant(_metric(map)))


def geometric_map(
    D: DomainMap | CellMap,
) -> tuple[AbstractCoordinateDomain, AbstractCoordinateDomain]:
    from .nodes import AbstractCoordinateDomain, CellMap, Mesh

    source, target = (
        (D.reference_cell, D.parent)
        if isinstance(D, CellMap)
        else (D.parameter_domain, D.ambient_domain)
    )
    if not isinstance(source, AbstractCoordinateDomain):
        raise TypeError("geometry requires a coordinate parameter domain")
    if isinstance(target, Mesh):
        return source, target.ambient_coordinate_domain
    if isinstance(target, AbstractCoordinateDomain):
        return source, target
    raise TypeError("geometry requires a coordinate map or mesh cell map")
