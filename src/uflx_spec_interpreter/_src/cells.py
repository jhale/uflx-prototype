from functools import reduce

from .arithmetic import add, divide, subtract, total
from .nodes import ComponentTensor, Coordinate, Index, Indexed, Integer, Tensor
from .tensors import component, inverse, matrix_product


def affine_map(F, points):
    # X -> p0 + sum over d < k of (p_(d+1) - p0) (E^-1 (X - V0))_d, on the reference cell F.
    k = F.dimension
    l = Index(("uflx.affine_map", F, points, 0))
    p0 = Indexed(points, (0, l))
    if k == 0:
        return ComponentTensor(p0, (l,))
    V = F.vertices
    Y = matrix_product(inverse(edges(F)), subtract(Coordinate(F), component(V, (0,))))
    terms = [subtract(Indexed(points, (d + 1, l)), p0) * Indexed(Y, (d,)) for d in range(k)]
    return ComponentTensor(total([p0, *terms]), (l,))


def edges(F):
    # The matrix E with the columns V_(d+1) - V_0.
    V, k = F.vertices, F.dimension
    return Tensor(
        tuple(
            Tensor(tuple(subtract(Indexed(V, (d + 1, r)), Indexed(V, (0, r))) for d in range(k)))
            for r in range(k)
        )
    )


def centroid(K):
    # The mean of the vertices of the reference cell K.
    V, m = K.vertices, K.num_vertices
    return divide(reduce(add, (component(V, (v,)) for v in range(m))), Integer(m))
