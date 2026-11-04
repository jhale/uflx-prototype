"""I01. Mesh integrals as sums over reference cells."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..algorithms.names import _derived_indices
from ..algorithms.substitution import _freshen
from ..arithmetic import _total, multiply
from ..geometry import density
from ..meshes import _cell_domain
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
from ..topology import cell_index

if TYPE_CHECKING:
    from ..nodes.base import AbstractExpression


__all__ = ["_mesh_integral", "_piece_integral"]


def _mesh_integral(e: AbstractExpression) -> AbstractExpression:
    # Expand a mesh integral into reference-cell integrals and internal cell sums.
    match e:
        case Integral(f, Mesh() as d) if f.value_shape:
            K = _derived_indices("uflx.expand_integrals", (f, d), len(f.value_shape))
            component_integral = Integral(Indexed(f, K), d)
            expanded = _mesh_integral(component_integral)
            return ComponentTensor(expanded, K)
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
    g = PullBack(f, piece if f.domain == D.ambient_domain else cells)
    reference_integrand = multiply(g, density(cells))
    term = _mesh_integral(Integral(reference_integrand, P))
    return IndexSum(term, (entry,))
