from __future__ import annotations

from typing import TYPE_CHECKING

from .nodes import DomainMap, PullBack

if TYPE_CHECKING:
    from .nodes.base import AbstractExpression
    from .nodes.domains import AbstractCoordinateDomain, CellMap, SideMap
    from .nodes.topology import AbstractTopology


def domain_map(
    P: AbstractCoordinateDomain | AbstractTopology,
    psi: AbstractExpression,
    Q: AbstractCoordinateDomain,
) -> DomainMap:
    return DomainMap(P, psi, Q)


def pull_back(f: AbstractExpression, D: DomainMap | CellMap | SideMap) -> PullBack:
    return PullBack(f, D)
