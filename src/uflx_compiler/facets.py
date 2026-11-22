"""Reference-facet tables in UFCx's reflection/rotation numbering."""

import basix
import numpy as np

from uflx import (
    EuclideanSpace,
    conormal,
    domain_map,
    jacobian,
    lower,
    pull_back,
    side_map,
)
from uflx import (
    mesh as make_mesh,
)
from uflx._src.algorithms.names import derived_index
from uflx._src.nodes import Conormal, SideMap
from uflx_basix.basix_cells import basix_cell
from uflx_basix.inclusions import affine_map, reference_normal
from uflx_basix.runtime import runtime_array
from uflx_basix.sides import outward_conormals

from .algebra import _normalizer
from .evaluation import evaluate
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


def _facet_tables(nodes, mesh, domain, width):
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
    F = basix_cell(facet_type)
    if domain.topology.cell_types != (F,):
        raise NotImplementedError("UFCx integration cell must be the parent reference facet type")
    replacements, bindings, inclusions, conormals = {}, {}, [], []
    for slot in range(width):
        facet = derived_index("compiler.ufcx.facet", slot)
        permutation = derived_index("compiler.ufcx.permutation", slot)
        positions = runtime_array(("ufcx.reference_facet", slot), tables.shape[-2:])
        bindings[facet] = f"entity_local_index[{slot}]"
        if width == 2:
            bindings[permutation] = f"quadrature_permutation[{slot}]"
            replacements[positions] = constant(tables)[facet, permutation]
        else:
            replacements[positions] = constant(tables[:, 0])[facet]
        A = domain_map(F, affine_map(F, positions), K)
        inclusions.append(A)
        conormals.append(reference_normal(A, positions, K))

    source = domain.topology
    maps = tuple(
        n
        for n in nodes
        if isinstance(n, SideMap)
        and n.parent in (mesh, mesh.topology)
        and n.source in (domain, source)
    )
    # This ABI binds complete reference facets at runtime, not arbitrary cuts.
    # Reject inclusions it cannot represent instead of silently changing them.
    normalize = _normalizer()
    for relation in maps:
        A = relation.entries[0][1]
        try:
            evaluate(normalize(lower(jacobian(A))), {})  # Must be reference-constant.
            points = np.asarray(
                [
                    evaluate(
                        normalize(
                            lower(
                                pull_back(
                                    A.parameterisation,
                                    domain_map(EuclideanSpace(0), F.vertices[v], F),
                                )
                            )
                        ),
                        {},
                    )
                    for v in range(F.num_vertices)
                ],
                dtype=float,
            )
        except (KeyError, TypeError, ValueError) as error:
            raise NotImplementedError(
                "UFCx requires affine, complete reference-facet inclusions"
            ) from error
        if not any(np.allclose(points, table) for table in tables.reshape(-1, *tables.shape[-2:])):
            raise NotImplementedError("UFCx requires complete reference-facet inclusions")
    geometry_map = domain.coordinate_function.map
    slot = geometry_map.entries[0][0]
    local_geometry = side_map(source, (mesh.topology[slot], inclusions[slot]))
    local_domain = make_mesh(
        pull_back(mesh.coordinate_function, local_geometry), domain.ambient_domain
    )
    mapping = {domain: local_domain}
    for relation in maps:
        slot = relation.entries[0][0]
        endpoint = local_domain if relation.source == domain else source
        mapping[relation] = side_map(endpoint, (relation.parent[slot], inclusions[slot]))

    # A normal's orientation is data, not part of the evaluation relation. The
    # UFCx ABI changes the selected facet at runtime; retain the declared sign.
    for node in nodes:
        if not isinstance(node, Conormal) or node.side_map not in maps:
            continue
        slot = node.side_map.entries[0][0]
        try:
            declared = np.asarray(
                evaluate(normalize(lower(node.reference_conormals[0])), {}), dtype=float
            )
            outward = np.asarray(
                evaluate(normalize(lower(outward_conormals(node.side_map)[0])), {}), dtype=float
            )
        except (KeyError, TypeError, ValueError) as error:
            raise NotImplementedError(
                "UFCx requires constant reference conormal orientations"
            ) from error
        scale = np.dot(declared, outward) / np.dot(outward, outward)
        if not np.isfinite(scale) or scale == 0 or not np.allclose(declared, scale * outward):
            raise ValueError("reference conormal must be nonzero and normal to the declared facet")
        mapping[node] = conormal(
            mapping[node.side_map], conormals[slot] if scale > 0 else -conormals[slot]
        )
    return local_domain, mapping, replacements, bindings
