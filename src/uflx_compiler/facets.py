"""Reference-facet tables in UFCx's reflection/rotation numbering."""

import basix
import numpy as np

from uflx_spec_interpreter._src.algorithms.names import derived_index
from uflx_toy.basix_cells import basix_cell
from uflx_toy.runtime import runtime_array
from uflx_toy.topology import tabulated_topology

from .quadrature import constant


def _permutations(cell_type):
    """Vertex orderings for permutation 2 * rotations + reflections."""
    if cell_type == basix.CellType.point:
        yield (0,)
        return
    vertices = basix.cell.geometry(cell_type)
    rotations = {
        basix.CellType.point: 1,
        basix.CellType.interval: 1,
        basix.CellType.triangle: 3,
        basix.CellType.quadrilateral: 4,
    }[cell_type]
    for rotation in range(rotations):
        for reflection in range(1 if cell_type == basix.CellType.point else 2):
            points = vertices.copy()
            for _ in range(rotation):
                x, y = points.T.copy()
                points[:, 0] = y
                points[:, 1] = 1 - x - y if cell_type == basix.CellType.triangle else 1 - x
            if reflection:
                points = 1 - points if cell_type == basix.CellType.interval else points[:, ::-1]
            yield tuple(int(np.argmin(np.linalg.norm(vertices - p, axis=1))) for p in points)


def _facet_tables(mesh, domain, width):
    """One local facet with separately selected reference maps for each side.

    Placeholders keep facet selection outside core lowering. Afterwards they
    become constant tables indexed by UFCx's entity and permutation inputs.
    """
    K = mesh.topology.cell_types[0]
    facets = basix.cell.topology(K.cell_type)[K.dimension - 1]
    types = {
        basix.cell.sub_entity_type(K.cell_type, K.dimension - 1, f) for f in range(len(facets))
    }
    if len(types) != 1:
        raise NotImplementedError("UFCx facets currently require a uniform reference facet type")
    (facet_type,) = types
    orders = tuple(_permutations(facet_type))
    vertices = basix.cell.geometry(K.cell_type)
    tables = np.array([[vertices[np.array(f)[list(order)]] for order in orders] for f in facets])
    names = ("+", "-") if width == 2 else (domain.coordinate_function.map.side,)
    replacements, bindings, sides, data = {}, {}, [], []
    for side, name in enumerate(names):
        facet = derived_index("compiler.ufcx.facet", side)
        permutation = derived_index("compiler.ufcx.permutation", side)
        positions = runtime_array(("ufcx.reference_facet", side), tables.shape[-2:])
        bindings[facet] = f"entity_local_index[{side}]"
        if width == 2:
            bindings[permutation] = f"quadrature_permutation[{side}]"
            replacements[positions] = constant(tables)[facet, permutation]
        else:
            replacements[positions] = constant(tables[:, 0])[facet]
        sides.append((name, mesh.topology, side))
        data.append(positions)
    topology = tabulated_topology(
        "ufcx_local_facet", (basix_cell(facet_type),), (tuple(sides),), (tuple(data),)
    )
    return topology, replacements, bindings
