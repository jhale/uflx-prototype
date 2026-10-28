from functools import reduce

from uflx_spec_interpreter import (
    absolute,
    add,
    cell_index,
    component_tensor,
    coordinate,
    determinant,
    divide,
    domain_map,
    gathered_index,
    index,
    indexed,
    inverse,
    jacobian,
    tensor,
)
from uflx_spec_interpreter._src.nodes import Integer


def affine_map(F, points):
    # X -> p0 + sum over d < k of (p_(d+1) - p0) (E^-1 (X - V0))_d, through the vertices of F.
    l = index(("toy.affine_map", F, points, 0))
    k, V = F.dimension, F.vertices
    if k == 0:
        return component_tensor(points[0, l], (l,))
    E = tensor(tuple(tensor(tuple(V[d + 1, r] - V[0, r] for d in range(k))) for r in range(k)))
    Y = inverse(E) @ (coordinate(F) - V[0])
    terms = [(points[d + 1, l] - points[0, l]) * Y[d] for d in range(k)]
    return component_tensor(reduce(add, terms, points[0, l]), (l,))


def side_tables(S, i, side, cells, positions, facets=None):
    # The inclusion through the positions of the vertices of each cell e, and its normal.
    # A cell that tabulates its facet normals reads the normal of the local facet of e.
    (_, T, j) = next(s for s in S.sides[i] if s[0] == side)
    F, K = S.cell_types[i], T.cell_types[j]
    e = cell_index(S, i)
    v, k = (index(("toy.inclusion", S, i, side, n)) for n in range(2))
    points = component_tensor(indexed(positions, (e, v, k)), (v, k))
    A = domain_map(F, affine_map(F, points), K)
    if F.dimension != K.dimension - 1:
        return cells, A, None
    if facets is not None:
        return cells, A, K.facet_normals[gathered_index(facets, (e,))]
    return cells, A, reference_normal(A, points, K)


def reference_normal(A, points, K):
    # The cofactor vector of the Jacobian of A, turned away from the centroid of K.
    J, n = jacobian(A), K.dimension

    def column_matrix(r):
        rows = (
            tensor((Integer(1 if q == r else 0), *(J[q, c] for c in range(n - 1))))
            for q in range(n)
        )
        return tensor(tuple(rows))

    m0 = tensor(tuple(determinant(column_matrix(r)) for r in range(n)))
    away = m0 @ (points[0] - centroid(K))
    return divide(away, absolute(away)) * m0


def centroid(K):
    V = K.vertices
    return divide(reduce(add, (V[v] for v in range(K.num_vertices))), Integer(K.num_vertices))


def vertex_rows(K, local):
    # The coordinates of the local vertices of K, row by row.
    return tensor(tuple(tensor(tuple(K.vertices[n, k] for k in range(K.dimension))) for n in local))
