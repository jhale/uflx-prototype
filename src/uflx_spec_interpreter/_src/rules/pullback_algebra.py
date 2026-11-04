"""PB01-PB06. Pointwise preservation, composition and retained pull-backs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..algorithms.rewriting import _rewrite
from ..algorithms.substitution import _freshen
from ..algorithms.traversal import _map_values
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
from ..nodes.static import _field_terminal
from .coordinates import _piola_law

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


__all__ = ["_composed", "_pointwise", "_pullback_remains"]


def _composed(
    phi: DomainMap,
    m: DomainMap | CellMap,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> DomainMap:
    # The map phi after m, so that only a pull-back by a cell keeps a pull-back inside it.
    # Traversal does not enter stored maps, so lower the parameterisation here.
    parameterisation = _rewrite(
        phi.parameterisation,
        lambda node: pb(node.expression, node.map) if isinstance(node, PullBack) else node,
    )
    composed_parameterisation = pb(parameterisation, m)
    return DomainMap(m.parameter_domain, composed_parameterisation, phi.ambient_domain)


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
        or (isinstance(f, PullBack) and isinstance(f.map, SideMap | CellMap))
        or (isinstance(f, Gradient) and not _is_cell(m))
        or (_piola_law(f) is not None and (not _is_cell(m) or m.ambient_domain != f.domain))
    )
