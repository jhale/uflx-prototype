"""Library constructors for explicit relations and separately chosen orientation."""

from uflx import (
    EuclideanSpace,
    conormal,
    domain_map,
    expand_pull_backs,
    pull_back,
    side_map,
    tensor,
)
from uflx.nodes import Mesh

from .inclusions import affine_map, reference_normal, vertex_rows


def topology(endpoint):
    return endpoint.topology if isinstance(endpoint, Mesh) else endpoint


def static_side_map(parent, source, *parent_slots):
    """Match source vertices to the explicitly selected parent cells."""
    T, S = topology(parent), topology(source)
    entries = []
    for i, j in enumerate(parent_slots):
        K, F = T.cell_types[j], S.cell_types[i]
        local = tuple(T.cells[j].index(v) for v in S.cells[i])
        entries.append((parent[j], domain_map(F, affine_map(F, vertex_rows(K, local)), K)))
    return side_map(source, *entries)


def tabulated_side_map(parent, source, *entries):
    """Build inclusions from (parent_slot, reference_vertex_positions) entries."""
    T, S = topology(parent), topology(source)
    return side_map(
        source,
        *(
            (parent[j], domain_map(F, affine_map(F, positions), T.cell_types[j]))
            for F, (j, positions) in zip(S.cell_types, entries, strict=True)
        ),
    )


def outward_conormals(relation):
    """Choose reference conormals away from convex reference-cell centroids.

    This policy is for planar affine facets. It is independent of map identity.
    """
    # This library constructor uses only public passes; evaluate the inclusion at
    # the first reference vertex to choose its outward direction.
    result = []
    for _, A in relation.entries:
        F, K = A.parameter_domain, A.ambient_domain
        point = domain_map(EuclideanSpace(0), F.vertices[0], F)
        position = expand_pull_backs(pull_back(A.parameterisation, point))
        result.append(reference_normal(A, tensor((position,)), K))
    return tuple(result)


def outward_conormal(relation):
    return conormal(relation, *outward_conormals(relation))
