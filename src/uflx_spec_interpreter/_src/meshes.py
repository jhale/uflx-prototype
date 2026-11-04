from __future__ import annotations

from typing import TYPE_CHECKING, Literal, TypeGuard, cast

from uflx_spec_interpreter._src.nodes import (
    AbstractTopology,
    CellMap,
    GatheredIndex,
    Mesh,
    Normal,
    PullBack,
    PushForward,
    SideMap,
    TransformationLaw,
)
from uflx_spec_interpreter._src.substitution import _substitute
from uflx_spec_interpreter._src.topology import cell_index

if TYPE_CHECKING:
    from uflx_spec_interpreter._src.nodes.base import AbstractExpression
    from uflx_spec_interpreter._src.nodes.indices import Index

    from .nodes.domains import AbstractCoordinateDomain
    from .nodes.maps import DomainMap


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = [
    "_cell_domain",
    "_cell_of",
    "_inclusion",
    "_is_cell",
    "cell_map",
    "image",
    "normal",
    "push_forward",
    "side_domain",
    "side_map",
    "transformation_law",
]


def _cell_domain(
    D: AbstractTopology | Mesh, j: int, e: int | Index | GatheredIndex | None = None
) -> CellMap:
    # Private compatibility spelling; both topology and mesh cells are CellMaps.
    return cell_map(D, j, e)


def cell_map(
    X: AbstractTopology | Mesh, j: int, e: int | Index | GatheredIndex | None = None
) -> CellMap:
    T = X.topology if isinstance(X, Mesh) else X
    return CellMap(X, j, cell_index(T, j) if e is None else e)


def image(x: AbstractExpression, Q: AbstractCoordinateDomain) -> Mesh:
    return Mesh(x, Q)


def transformation_law(
    variances: tuple[Literal["covariant", "contravariant"], ...], weight: int
) -> TransformationLaw:
    return TransformationLaw(variances, weight)


def push_forward(f: AbstractExpression, D: Mesh, m: TransformationLaw) -> PushForward:
    return PushForward(f, D, m)


def _cell_of(m: DomainMap | CellMap | SideMap) -> tuple[int, int | Index | GatheredIndex] | None:
    return (
        (m.cell_group, m.entry)
        if isinstance(m, CellMap) and isinstance(m.ambient_domain, Mesh)
        else None
    )


def _is_cell(m: DomainMap | CellMap | SideMap) -> TypeGuard[CellMap]:
    return _cell_of(m) is not None


def side_map(X: Mesh | AbstractTopology, P: Mesh | AbstractTopology, s: str) -> SideMap:
    return SideMap(X, P, s)


def side_domain(D: Mesh, S: AbstractTopology, s: str) -> Mesh:
    x = D.coordinate_function
    return image(PullBack(x, side_map(D.topology, S, s)), D.ambient_domain)


def _inclusion(
    X: Mesh | AbstractTopology, S: AbstractTopology, i: int, s: str, e: int | Index | GatheredIndex
) -> tuple[CellMap, DomainMap]:
    # B, the cell on the side s of cell e of group i, and A, the inclusion of cell e into it.
    (_, _, j) = next(side for side in S.sides[i] if side[0] == s)
    cells, A, _ = S.side_tables(i, s)
    c = cell_index(S, i)
    g = GatheredIndex(cells, (e,))
    B = CellMap(X, j, g)
    return B, cast("DomainMap", _substitute(A, {c: e}))


def normal(sm: SideMap) -> Normal:
    return Normal(sm)
