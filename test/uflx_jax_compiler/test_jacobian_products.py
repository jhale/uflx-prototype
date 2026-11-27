"""Geometry identities in the compiler's lowering of mapped forms."""

import basix
import jax
import numpy as np
import pytest

from uflx._src.algorithms.traversal import preorder
from uflx.calculus import coordinate, directional_derivative, integral
from uflx.forms import function
from uflx.geometry import create_mesh, euclidean_space, jacobian
from uflx.nodes import Inverse
from uflx.passes import lower as reference_lower
from uflx.tensors import index, index_sum, inverse
from uflx_basix.cells import create_topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space
from uflx_jax_compiler import trace
from uflx_jax_compiler._jacobian_products import cancel_jacobian_products
from uflx_jax_compiler.passes import lower


def has_inverse(expr):
    return any(isinstance(node, Inverse) for node in preorder(expr))


@pytest.mark.parametrize(
    ("cell", "gdim"),
    [(basix.CellType.triangle, 2), (basix.CellType.triangle, 3), (basix.CellType.tetrahedron, 3)],
)
def test_contraction_from_piola_pullbacks(cell, gdim):
    topology = create_topology(cell)
    element = basix.create_element(basix.ElementFamily.P, cell, 2)
    X = function_space(topology, element, shape=(gdim,))
    x, h = (function(X, name) for name in ("x", "h"))
    mesh = create_mesh(x, euclidean_space(gdim))
    U, V = (
        function_space(mesh, basix.create_element(family, cell, 1))
        for family in (basix.ElementFamily.RT, basix.ElementFamily.N1E)
    )
    u, v = function(U, "u"), function(V, "v")
    energy = integral(coordinate(mesh)[0] * (u @ v), mesh)
    shape = directional_derivative(energy, x, h)
    reduced = lower(energy)
    derivative = lower(shape)
    assert has_inverse(reference_lower(energy))
    assert not has_inverse(reduced)
    assert not has_inverse(derivative)
    assert not has_inverse(lower(integral(v @ u, mesh)))
    assert has_inverse(lower(integral(v @ v, mesh)))

    rule = basix.make_quadrature(cell, 5)

    def kernel(expr, *fields):
        return jax.jit(trace(apply_quadrature(expr, *rule), *fields))

    oracle = kernel(reference_lower(energy), x, u, v)
    primal = kernel(reduced, x, u, v)
    tangent = kernel(derivative, x, h, u, v)
    rng = np.random.default_rng(653)
    coords = np.zeros((element.dim, gdim))
    coords[:, : element.points.shape[1]] = element.points
    coords += 0.025 * rng.normal(size=coords.shape)
    direction = 0.15 * rng.normal(size=coords.shape)
    dofs = [rng.normal(size=space.reference_space.num_dofs) for space in (U, V)]
    for sign in (1, -1):
        points = coords.copy()
        points[:, 0] *= sign
        np.testing.assert_allclose(primal(points, *dofs), oracle(points, *dofs), atol=1e-12)
        step = 1e-5
        fd = (
            oracle(points + step * direction, *dofs) - oracle(points - step * direction, *dofs)
        ) / (2 * step)
        np.testing.assert_allclose(tangent(points, direction, *dofs), fd, rtol=1e-8, atol=1e-10)


def test_only_matching_cell_maps_and_contracted_indices_cancel():
    cell = basix.CellType.triangle
    X = function_space(
        create_topology(cell), basix.create_element(basix.ElementFamily.P, cell, 1), shape=(3,)
    )
    x, y = (function(X, key) for key in ("x", "y"))
    J, L = (jacobian(create_mesh(f, euclidean_space(3))[0]) for f in (x, y))
    K = inverse(J)
    i = index("physical")
    a = index("reference")

    def cancel(expr):
        return cancel_jacobian_products(expr, {J, L})

    assert not has_inverse(cancel(K @ J))
    # On a surface J K is the tangent-plane projector.
    assert cancel(J @ K) == J @ K
    assert cancel(K @ L) == K @ L
    weighted = index_sum(K[a, i] * J[i, 0] * J[i, 1], (i,))
    assert cancel(weighted) == weighted
