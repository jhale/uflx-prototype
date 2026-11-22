"""I01. Mesh integrals as sums over reference cells."""

from __future__ import annotations

from typing import TYPE_CHECKING

from ..arithmetic import multiply, total
from ..geometry import density
from ..meshes import cell_map
from ..nodes import DomainMap, Integral, Mesh, PullBack

if TYPE_CHECKING:
    from ..nodes.expressions import AbstractExpression


def mesh_integral(e: AbstractExpression) -> AbstractExpression:
    # Expand a mesh integral into reference-cell integrals over fixed local slots.
    match e:
        case Integral(f, Mesh() as d):
            return total(reference_integral(f, d, j) for j in range(len(d.topology.cell_types)))
        case _:
            return e


def reference_integral(f: AbstractExpression, D: Mesh, cell_slot: int) -> Integral:
    """Express one local mesh cell's integral on its reference cell."""
    cell = cell_map(D, cell_slot)
    P = cell.reference_cell
    piece = DomainMap(P, cell.parameterisation, D.ambient_domain)
    g = PullBack(f, piece if f.domain == D.ambient_domain else cell)
    return Integral(multiply(g, density(cell)), P)
