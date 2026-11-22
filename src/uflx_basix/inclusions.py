from functools import reduce

from uflx import (
    absolute,
    add,
    component_tensor,
    coordinate,
    determinant,
    divide,
    index,
    inverse,
    jacobian,
    tensor,
)
from uflx.nodes import Integer


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
