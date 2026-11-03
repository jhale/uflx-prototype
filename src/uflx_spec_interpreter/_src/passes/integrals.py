from __future__ import annotations

from typing import TYPE_CHECKING

from ..arithmetic import _total, multiply
from ..domains import _cell_domain
from ..geometry import density
from ..names import _derived_indices
from ..nodes import (
    ComponentTensor,
    DomainMap,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Mesh,
    PullBack,
)
from ..static import _shape, _support
from ..substitution import _bottom_up, _everywhere, _freshen
from ..topology import cell_index

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


# Shared implementation helpers; the package public API is declared in __init__.py.
__all__ = ["_in_n4", "expand_integrals"]


def _rule(e: AbstractExpression) -> AbstractExpression:
    # Expand a mesh integral into reference-cell integrals and internal cell sums.
    match e:
        case Integral(f, Mesh() as d) if _shape(f):
            K = _derived_indices("uflx.expand_integrals", (f, d), len(_shape(f)))
            return ComponentTensor(_rule(Integral(Indexed(f, K), d)), K)
        case Integral(f, Mesh() as d):
            return _total(_piece_integral(f, d, j) for j in range(len(d.topology.cell_types)))
        case _:
            return e


def _piece_integral(f: AbstractExpression, D: Mesh, j: int) -> AbstractExpression:
    # Pulled back by the cells of D, or by the piece if the support is the ambient domain.
    cells = _cell_domain(D, j)
    c = cell_index(D.topology, j)
    # Reserve integrand names before inserting f under the internal cell binder.
    # Reuse the canonical cell index unless it would capture a name in f.
    template = IndexSum(PullBack(Integer(1), cells), (c,))
    (entry,) = _freshen(template, [f]).indices
    if entry is not c:
        cells = _cell_domain(D, j, entry)
    P = cells.parameter_domain
    piece = DomainMap(P, cells.parameterisation, D.ambient_domain)
    g = PullBack(f, piece if _support(f) == D.ambient_domain else cells)
    term = _rule(Integral(multiply(g, density(cells)), P))
    return IndexSum(term, (entry,))


def expand_integrals(f: AbstractExpression) -> AbstractExpression:
    # Inner integrals first; integrand indices remain free.
    return _bottom_up(f, _rule)


def _in_n4(e: AbstractExpression) -> bool:
    def holds(n: AbstractExpression) -> bool:
        match n:
            case Integral(_, Mesh()):
                return False
            case _:
                return True

    return _everywhere(e, holds)
