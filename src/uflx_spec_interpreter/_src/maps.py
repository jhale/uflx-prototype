from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import DomainMap, PullBack

if TYPE_CHECKING:
    from .nodes.domains import AbstractCoordinateDomain
    from .nodes.expressions import AbstractExpression
    from .nodes.meshes import CellMap, SideMap
    from .nodes.topology import AbstractTopology


def domain_map(
    parameter_domain: AbstractCoordinateDomain | AbstractTopology,
    parameterisation: AbstractExpression,
    ambient_domain: AbstractCoordinateDomain,
) -> DomainMap:
    return DomainMap(parameter_domain, parameterisation, ambient_domain)


def pull_back(expression: AbstractExpression, map: DomainMap | CellMap | SideMap) -> PullBack:
    return PullBack(expression, map)
