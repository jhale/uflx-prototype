import numpy as np

from uflx.geometry import domain_map
from uflx_basix.cells import reference_simplex
from uflx_basix.inclusions import affine_map, centroid, reference_normal
from uflx_oracle import at, evaluate

from ..uflx_basix.test_cells import table


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


def test_the_reference_normal_of_a_face_of_the_tetrahedron_points_out():
    # The face with the vertices (1, 2, 3), orthogonal to the face and out of the tetrahedron.
    K, F = reference_simplex(3), reference_simplex(2)
    points = table([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    A = domain_map(F, affine_map(F, points), K)
    m = value(reference_normal(A, points, K), F, [0.2, 0.3])
    assert np.allclose(m / np.linalg.norm(m), np.ones(3) / np.sqrt(3))


def test_the_centroid_of_the_tetrahedron():
    assert np.allclose(value(centroid(reference_simplex(3))), (0.25, 0.25, 0.25))
