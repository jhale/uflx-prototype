# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Analytic checks of the reference interpreter, independent of lowering and compilation."""

import numpy as np
import pytest

import uflx_jax_compiler
from uflx.arithmetic import num
from uflx.calculus import coordinate, curl, divergence, gradient, integral
from uflx.forms import apply, form, function
from uflx.geometry import pull_back
from uflx.nodes import EuclideanSpace
from uflx.tensors import index, tensor
from uflx_jax_compiler import reference_trace

from ..uflx.spaces import space as field_space
from ..uflx_basix.cells import reference_simplex


def test_evaluation_reads_current_bindings_and_form_arguments():
    V = field_space("coefficient", None, scalar_type="real")
    u = function(V, "u")
    kernel = uflx_jax_compiler.trace(u + u, u)
    assert kernel(lambda: 3) == 6
    assert kernel(4) == 8
    v = function(V, "v")
    assert uflx_jax_compiler.trace(u + apply(form(v, (v,)), (u,)), u)(3) == 6
    action = uflx_jax_compiler.trace(form(u + u, (u,)))
    assert action(5) == 10
    with pytest.raises(TypeError, match="expected 1 positional value"):
        action()


def test_cell_point_and_free_index_are_explicit_inputs():
    from ..uflx.spaces import space as field_space
    from ..uflx_basix.integrals import SQUARE

    V = field_space("vector", SQUARE, (2,))
    u = function(V, "field")
    i = index("component")
    point = [0.2, 0.3]
    fields = {0: lambda x: -x, 1: lambda x: 2 * x}
    sample = uflx_jax_compiler.trace(u[i], u, i, SQUARE)
    with pytest.raises(ValueError, match="local cell slot"):
        sample(fields, 1, point)
    for slot, expected in enumerate((-0.3, 0.6)):
        piece = SQUARE[slot]
        local = uflx_jax_compiler.trace(pull_back(u[i], piece), u, i, piece.reference_cell)
        assert local(fields, 1, point) == pytest.approx(expected)
        # An assembler can also bind the selected field directly.
        assert sample(fields[slot], 1, point) == pytest.approx(expected)


def test_callback_reentry_uses_its_explicit_point():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    u = function(field_space("field", P, scalar_type="real"), "u")

    def shifted(point):
        with pytest.raises(KeyError):
            uflx_jax_compiler.trace(x)()
        return uflx_jax_compiler.trace(x, P)(point + 1)

    assert uflx_jax_compiler.trace(u + x, u, P)(shifted, [2]) == 5


@pytest.mark.parametrize("dimension", [2, 3])
def test_vector_calculus_against_an_affine_field(dimension):
    P = EuclideanSpace(dimension)
    x = coordinate(P)
    components = (x[0] + num(2) * x[1], num(3) * x[0] - x[1])
    vector = tensor(components if dimension == 2 else (*components, x[2]))
    point = np.zeros(dimension)
    assert reference_trace(divergence(vector, P), P)(point) == pytest.approx(dimension - 2)
    np.testing.assert_allclose(
        reference_trace(curl(vector, P), P)(point), 1 if dimension == 2 else [0, 0, 1]
    )


def test_point_gradient_and_tetrahedron_quadrature():
    P = EuclideanSpace(0)
    assert uflx_jax_compiler.trace(gradient(num(1), P), P)([]).shape == (0,)
    K = reference_simplex(3)
    volume = integral(num(1), K)
    moment = integral(coordinate(K)[2], K)
    rule = K.quadrature(1)
    assert uflx_jax_compiler.trace(volume, volume)(rule) == pytest.approx(1 / 6)
    assert uflx_jax_compiler.trace(moment, moment)(rule) == pytest.approx(1 / 24)


def test_unbounded_euclidean_space_has_no_unit_simplex_quadrature():
    with pytest.raises(ValueError, match="explicit quadrature"):
        uflx_jax_compiler.trace(integral(num(1), EuclideanSpace(2)))


@pytest.mark.parametrize("ambient_dimension", [2, 3])
def test_area_and_mapped_gradient_on_a_thin_triangle(ambient_dimension):
    from uflx.geometry import push_forward, transformation_law

    from ..uflx.library import static_topology
    from ..uflx_basix.meshes import vertex_mesh

    topology = static_topology((reference_simplex(2),), ((0, 1, 2),))
    height = 1e-9
    vertices = np.zeros((3, ambient_dimension))
    vertices[1:, 0] = 1
    vertices[2, 1] = height
    mesh, geometry = vertex_mesh(topology, vertices)
    xi = function(field_space("reference coordinate", topology, scalar_type="real"), "xi")
    area = integral(num(1), mesh)
    measure = reference_trace(area, area, *geometry)
    rule = topology.cell_types[0].quadrature(1)
    np.testing.assert_allclose(measure(rule, *geometry.values()), height / 2, atol=0)
    bindings = geometry | {xi: lambda p: p[0]}

    # The first reference coordinate is x - y / height on the physical triangle.
    field = push_forward(xi, mesh, transformation_law((), 0))
    point = 0.2 * vertices[1] + 0.3 * vertices[2]
    assert reference_trace(field, *bindings, mesh)(*bindings.values(), point) == pytest.approx(0.2)
    expected = (1, -1 / height, 0)[:ambient_dimension]
    actual = reference_trace(gradient(field, mesh), *bindings, mesh)(*bindings.values(), point)
    np.testing.assert_allclose(actual, expected, atol=0)


@pytest.mark.parametrize("cell_name", ["prism", "pyramid"])
def test_basix_vertex_field_reproduces_reference_coordinates(cell_name):
    import basix

    from uflx_basix.cells import cell

    from ..uflx.library import static_topology

    cell_type = basix.CellType[cell_name]
    K = cell(cell_type)
    vertices = tuple(tuple(map(float, vertex)) for vertex in basix.geometry(cell_type))
    topology = static_topology((K,), (tuple(range(len(vertices))),))
    from ..uflx_basix.meshes import vertex_mesh

    domain, geometry = vertex_mesh(topology, vertices)
    field = domain.coordinate_function
    point = [0.2, 0.1, 0.3]
    np.testing.assert_allclose(
        uflx_jax_compiler.trace(field, *geometry, topology)(*geometry.values(), point), point
    )


def test_core_interpretation_without_basix_or_lowering():
    import subprocess
    import sys

    code = """
import sys

class BlockDependencies:
    def find_spec(self, name, path=None, target=None):
        if name == "basix":
            raise ModuleNotFoundError("optional Basix dependency", name=name)
        if name.startswith("uflx._src.passes"):
            raise AssertionError(name)

sys.meta_path.insert(0, BlockDependencies())
from uflx.arithmetic import num
import uflx_jax_compiler
from uflx_jax_compiler import reference_trace
assert uflx_jax_compiler.trace(num(2) * num(3))() == 6
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_basix_form_slot_accepts_an_applied_field():
    import basix

    from uflx_basix.cells import cell
    from uflx_basix.spaces import local_space

    K = cell(basix.CellType.triangle)
    V = local_space(K, basix.create_element(basix.ElementFamily.P, K.cell_type, 1))
    u = function(V, "u")
    identity = form(u, (u,))
    field = coordinate(K)[0]
    point = [0.2, 0.3]
    assert uflx_jax_compiler.trace(apply(identity, (field,)), K)(point) == pytest.approx(0.2)
    w = function(V, "w")
    assert uflx_jax_compiler.trace(apply(identity, (w,)), w, K)(
        lambda x: x[1], point
    ) == pytest.approx(0.3)


def test_applied_field_reads_the_cell_where_the_form_samples_it():
    from ..uflx_basix.integrals import GEOMETRY, D

    V = field_space("piecewise", D, scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    mass = integral(v, D)
    expression = apply(form(mass, (v,)), (u,))

    # The unit square consists of two triangles of area 1/2.
    def sample(point):
        return uflx_jax_compiler.trace(coordinate(D)[0], D)(point)

    fields = {0: sample, 1: lambda point: sample(point) + 2}
    kernel = reference_trace(expression, mass, u, *GEOMETRY)
    assert kernel(
        D.topology.cell_types[0].quadrature(1), fields, *GEOMETRY.values()
    ) == pytest.approx(1.5)


def test_surface_gradient_uses_the_supplied_slot_on_a_shared_edge():
    from ..uflx_basix.surface import GEOMETRY, M

    # The two triangles meet at this point but have different tangent planes.
    points = {M: [0.5, 0.5, 0.25]}
    expr = gradient(coordinate(M)[2], M)
    with pytest.raises(ValueError, match="local cell slot"):
        uflx_jax_compiler.trace(expr, *GEOMETRY, M)(*GEOMETRY.values(), points[M])
    for slot, expected in enumerate((np.array([0, 2, 1]) / 5, np.array([-2, 4, 5]) / 9)):
        piece = M[slot]
        reference_point = ([0, 0.5], [0.5, 0])[slot]
        sample = reference_trace(pull_back(expr, piece), *GEOMETRY, piece.reference_cell)
        actual = sample(*GEOMETRY.values(), reference_point)
        np.testing.assert_allclose(actual, expected, atol=1e-14)
