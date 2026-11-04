"""PB01-PB06. Pointwise preservation, composition and retained pull-backs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..meshes import _is_cell
from ..nodes import (
    Action,
    CellMap,
    DomainMap,
    Form,
    Gradient,
    PullBack,
    PushForward,
    SideMap,
)
from ..rewriting import _rewrite
from ..rules.coordinates import _piola_law
from ..static import _field_terminal
from ..substitution import _freshen
from ..traversal import _map_values

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


__all__ = ["_composed", "_pointwise", "_pullback_remains"]


def _kept(m: DomainMap | CellMap | SideMap) -> bool:
    # a pull-back stays inside another only by a cell, a cell map or a side map.
    return isinstance(m, SideMap | CellMap) or _is_cell(m)


def _composed(
    phi: DomainMap,
    m: DomainMap | CellMap,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> DomainMap:
    # The map phi after m, so that only a pull-back by a cell keeps a pull-back inside it.
    psi = pb(_lowered_copy(pb, phi.parameterisation), m)
    return DomainMap(m.parameter_domain, psi, phi.ambient_domain)


def _lowered_copy(
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
    e: AbstractExpression,
) -> AbstractExpression:
    # The parameterisation of a map that a row builds is lowered as the row builds it, because
    # a lowering pass does not enter maps.
    return _rewrite(e, lambda n: pb(n.expression, n.map) if isinstance(n, PullBack) else n)


def _pointwise(
    f: AbstractExpression,
    m: DomainMap | CellMap | SideMap,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> AbstractExpression:
    # a binding node is made fresh for the free indices of the map, which occur in the map.
    fresh = _freshen(f, [m])
    if isinstance(fresh, Form):
        return Form(pb(fresh.expression, m), fresh.arguments)
    return _map_values(fresh, lambda x: pb(x, m), fresh.value_fields)


def _pullback_remains(f: AbstractExpression, m: DomainMap | CellMap | SideMap) -> bool:
    # The pull-backs that no row of expand_pull_backs rewrites, which a library reads.
    return not isinstance(f, Action | Form) and (
        isinstance(m, SideMap)
        or _field_terminal(f)
        or isinstance(f, PushForward)
        or (isinstance(f, PullBack) and _kept(f.map))
        or (isinstance(f, Gradient) and not _is_cell(m))
        or (_piola_law(f) is not None and (not _is_cell(m) or m.ambient_domain != f.domain))
    )
