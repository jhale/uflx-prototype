"""Interior facet geometry for the mixed-cell DG example."""

from uflx.geometry import conormal, side_domain, side_map
from uflx_basix.cells import topology
from uflx_basix.facets import reference_facet


def interior_facet(mesh, local_facets, permutations):
    """Build the shared domain, its two restrictions, and the physical plus normal."""
    plus_cell = mesh[0]
    minus_cell = mesh[1]
    into_plus, ref_normal = reference_facet(
        plus_cell.reference_cell, local_facets[0], permutations[0]
    )
    into_minus, _ = reference_facet(minus_cell.reference_cell, local_facets[1], permutations[1])

    edge_topology = topology(into_plus.parameter_domain.cell_type)
    edge = side_domain(mesh, edge_topology, (0, into_plus))
    plus = side_map(edge, (plus_cell, into_plus))
    minus = side_map(edge, (minus_cell, into_minus))
    return edge, plus, minus, conormal(plus, ref_normal)
