import numpy as np

from uflx_oracle import at, evaluate
from uflx_spec_interpreter import affine_map, num, tensor
from uflx_spec_interpreter._src.cells import affine_jacobian, centroid
from uflx_spec_interpreter._src.passes.pullbacks import cofactor_normal
from uflx_toy.cells import reference_simplex


def table(rows):
    return tensor(tuple(tensor(tuple(num(x) for x in row)) for row in rows))


def value(e, P=None, X=()):
    return np.real(at(P, np.asarray(X, dtype=float), lambda: evaluate(e, {})(())))


def test_the_hypotenuse_of_the_unit_triangle():
    # The sub-entity with the vertex numbers (1, 2), with psi(t) = (1 - t, t).
    I = reference_simplex(1)
    psi = affine_map(I, table([(1, 0), (0, 1)]))
    assert psi.value_shape == (2,)
    for t in (0.0, 0.3, 1.0):
        assert np.allclose(value(psi, I, [t]), (1 - t, t))


def test_the_end_point_of_the_unit_interval():
    assert np.allclose(value(affine_map(reference_simplex(0), table([(1,)]))), (1,))


def test_the_cofactor_normal_of_a_face_of_the_tetrahedron():
    # The face with the vertices (1, 2, 3), which the old triangle-only normal could not build.
    F = reference_simplex(2)
    points = table([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    J = affine_jacobian(F, points)
    m = value(cofactor_normal(J, 3))
    assert np.allclose(value(J).T @ m, 0) and np.allclose(np.abs(m), np.abs(m[0])) and m[0] != 0


def test_the_centroid_of_the_tetrahedron():
    assert np.allclose(value(centroid(reference_simplex(3))), (0.25, 0.25, 0.25))
