from uflx.arithmetic import num
from uflx.calculus import coordinate
from uflx.tensors import component_tensor, index, inverse, tensor

from ..uflx.inclusions import affine_map
from .cells import reference_simplex

# The affine maps and normals of the library of the toy, which side tables are built from.


def table(rows):
    return tensor(tuple(tensor(tuple(num(x) for x in row)) for row in rows))


def test_the_affine_map_is_the_composition_of_the_example():
    F = reference_simplex(2)
    points = table([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    l = index(("toy.affine_map", F, points, 0))
    V = F.vertices
    E = tensor(tuple(tensor(tuple(V[d + 1, r] - V[0, r] for d in range(2))) for r in range(2)))
    Y = inverse(E) @ (coordinate(F) - V[0])
    terms = (
        points[0, l] + (points[1, l] - points[0, l]) * Y[0] + (points[2, l] - points[0, l]) * Y[1]
    )
    assert affine_map(F, points) == component_tensor(terms, (l,))
    P = reference_simplex(0)
    m = index(("toy.affine_map", P, points, 0))
    assert affine_map(P, points) == component_tensor(points[0, m], (m,))
