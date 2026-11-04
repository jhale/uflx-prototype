"""I01. Mesh integrals as sums over reference cells."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..algorithms.names import derived_indices
from ..algorithms.substitution import freshen
from ..arithmetic import multiply, total
from ..geometry import density
from ..meshes import cell_map
from ..nodes import (
    ComponentTensor,
    DomainMap,
    GatheredIndex,
    Index,
    Indexed,
    IndexSum,
    Integer,
    Integral,
    Mesh,
    PullBack,
)
from ..topology import cell_index

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def mesh_integral(e: AbstractExpression) -> AbstractExpression:
    # Expand a mesh integral into reference-cell integrals and internal cell sums.
    match e:
        case Integral(f, Mesh() as d) if f.value_shape:
            K = derived_indices("uflx.expand_integrals", (f, d), len(f.value_shape))
            component_integral = Integral(Indexed(f, K), d)
            expanded = mesh_integral(component_integral)
            return ComponentTensor(expanded, K)
        case Integral(f, Mesh() as d):
            return total(_piece_integral(f, d, j) for j in range(len(d.topology.cell_types)))
        case _:
            return e


def _piece_integral(f: AbstractExpression, D: Mesh, j: int) -> AbstractExpression:
    # Pulled back by the cells of D, or by the piece if the support is the ambient domain.
    cells = cell_map(D, j)
    c = cell_index(D.topology, j)
    # Reserve integrand names before inserting f under the internal cell binder.
    # Reuse the canonical cell index unless it would capture a name in f.
    template = IndexSum(PullBack(Integer(1), cells), (c,))
    (entry,) = freshen(template, [f]).indices
    return IndexSum(reference_integral(f, D, j, entry), (entry,))


def reference_integral(
    f: AbstractExpression, D: Mesh, group: int, entry: int | Index | GatheredIndex
) -> Integral:
    """Express one selected mesh entity's integral on its reference cell.

    The entry is supplied by the caller; this operation introduces no cell sum.
    """
    cells = cell_map(D, group, entry)
    P = cells.parameter_domain
    piece = DomainMap(P, cells.parameterisation, D.ambient_domain)
    g = PullBack(f, piece if f.domain == D.ambient_domain else cells)
    return Integral(multiply(g, density(cells)), P)
