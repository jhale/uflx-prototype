"""PB01-PB06. Pointwise preservation, composition and retained pull-backs."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING

from ..algorithms.rewriting import rewrite
from ..algorithms.substitution import freshen
from ..algorithms.traversal import map_values
from ..meshes import is_cell
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
from ..nodes.static import field_terminal
from .coordinates import piola_law

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def composed(
    phi: DomainMap,
    m: DomainMap | CellMap,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> DomainMap:
    # The map phi after m, so that only a pull-back by a cell keeps a pull-back inside it.
    # Traversal does not enter stored maps, so lower the parameterisation here.
    parameterisation = rewrite(
        phi.parameterisation,
        lambda node: pb(node.expression, node.map) if isinstance(node, PullBack) else node,
    )
    composed_parameterisation = pb(parameterisation, m)
    return DomainMap(
        m.reference_cell if isinstance(m, CellMap) else m.parameter_domain,
        composed_parameterisation,
        phi.ambient_domain,
    )


def pointwise(
    f: AbstractExpression,
    m: DomainMap | CellMap | SideMap,
    pb: Callable[[AbstractExpression, DomainMap | CellMap | SideMap], AbstractExpression],
) -> AbstractExpression:
    # a binding node is made fresh for the free indices of the map, which occur in the map.
    fresh = freshen(f, [m])
    if isinstance(fresh, Form):
        return Form(pb(fresh.expression, m), fresh.arguments)
    return map_values(fresh, lambda x: pb(x, m), fresh.value_fields)


def pullback_remains(f: AbstractExpression, m: DomainMap | CellMap | SideMap) -> bool:
    # The pull-backs that no row of expand_pull_backs rewrites, which a library reads.
    return not isinstance(f, Action | Form) and (
        isinstance(m, SideMap)
        or field_terminal(f)
        or isinstance(f, PushForward)
        or (isinstance(f, PullBack) and isinstance(f.map, SideMap | CellMap))
        or (isinstance(f, Gradient) and not is_cell(m))
        or (piola_law(f) is not None and (not is_cell(m) or m.parent != f.domain))
    )
