from __future__ import annotations

from typing import TYPE_CHECKING

from ..arithmetic import multiply, total
from ..domains import cell_domain
from ..geometry import density
from ..names import derived_indices
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
from ..static import shape, support
from ..substitution import bottom_up, everywhere, freshen
from ..topology import cell_index

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


def rule(e: AbstractExpression) -> AbstractExpression:
    # Expand a mesh integral into reference-cell integrals and internal cell sums.
    match e:
        case Integral(f, Mesh() as d) if shape(f):
            K = derived_indices("uflx.expand_integrals", (f, d), len(shape(f)))
            return ComponentTensor(rule(Integral(Indexed(f, K), d)), K)
        case Integral(f, Mesh() as d):
            return total(piece_integral(f, d, j) for j in range(len(d.topology.cell_types)))
        case _:
            return e


def piece_integral(f: AbstractExpression, D: Mesh, j: int) -> AbstractExpression:
    # Pulled back by the cells of D, or by the piece if the support is the ambient domain.
    cells = cell_domain(D, j)
    c = cell_index(D.topology, j)
    # Reserve integrand names before inserting f under the internal cell binder.
    # Reuse the canonical cell index unless it would capture a name in f.
    template = IndexSum(PullBack(Integer(1), cells), (c,))
    (entry,) = freshen(template, [f]).indices
    if entry is not c:
        cells = cell_domain(D, j, entry)
    P = cells.parameter_domain
    piece = DomainMap(P, cells.parameterisation, D.ambient_domain)
    g = PullBack(f, piece if support(f) == D.ambient_domain else cells)
    term = rule(Integral(multiply(g, density(cells)), P))
    return IndexSum(term, (entry,))


def expand_integrals(f: AbstractExpression) -> AbstractExpression:
    # Inner integrals first; integrand indices remain free.
    return bottom_up(f, rule)


def in_n4(e: AbstractExpression) -> bool:
    def holds(n: AbstractExpression) -> bool:
        match n:
            case Integral(_, Mesh()):
                return False
            case _:
                return True

    return everywhere(e, holds)
