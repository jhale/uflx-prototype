"""Analytic checks of the reference interpreter, independent of lowering and compilation."""

import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate, curl, divergence, gradient, integral
from uflx.forms import apply, form, function
from uflx.nodes import EuclideanSpace
from uflx.tensors import index, tensor
from uflx_basix.cells import reference_simplex
from uflx_oracle.interpreter import Interpreter

from ..uflx.spaces import space


def test_evaluation_reads_current_bindings_and_form_arguments():
    V = space("coefficient", scalar_type="real")
    u = function(V, "u")
    env = {u: lambda: 3}
    interpreter = Interpreter(env)
    expression = u + u
    assert interpreter.evaluate(expression) == 6
    v = function(V, "v")
    assert interpreter.evaluate(u + apply(form(v, (v,)), (u,))) == 6
    env[u] = 4
    assert interpreter.evaluate(expression) == 8
    assert interpreter.evaluate(form(expression, (u,)), arguments=(5,)) == 10
    with pytest.raises(ValueError, match="expected 1 form argument"):
        interpreter.evaluate(form(expression, (u,)))


def test_cell_point_and_free_index_are_explicit_inputs():
    from uflx_basix.spaces import space

    from ..uflx_basix.integrals import SQUARE

    V = space("vector", (2,), domain=SQUARE)
    u = function(V, "field")
    i = index("component")
    point = {SQUARE: [0.2, 0.3]}
    fields = {0: lambda x: -x, 1: lambda x: 2 * x}
    interpreter = Interpreter({u: fields})
    with pytest.raises(ValueError, match="local cell slot"):
        interpreter.evaluate(u[i], indices={i: 1}, points=point)
    for slot, expected in enumerate((-0.3, 0.6)):
        assert interpreter.evaluate(
            u[i], indices={i: 1}, points=point, cells={SQUARE: slot}
        ) == pytest.approx(expected)
        # An assembler can bind the selected field directly.
        local = Interpreter({u: fields[slot]})
        assert local.evaluate(u[i], indices={i: 1}, points=point) == pytest.approx(expected)


def test_callback_reentry_uses_its_explicit_point():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    u = function(space("field", domain=P, scalar_type="real"), "u")

    def shifted(point):
        with pytest.raises(KeyError):
            interpreter.evaluate(x)
        return interpreter.evaluate(x, points={P: point + 1})

    interpreter = Interpreter({u: shifted})
    assert interpreter.evaluate(u + x, points={P: [2]}) == 5


@pytest.mark.parametrize("dimension", [2, 3])
def test_vector_calculus_against_an_affine_field(dimension):
    P = EuclideanSpace(dimension)
    x = coordinate(P)
    components = (x[0] + num(2) * x[1], num(3) * x[0] - x[1])
    vector = tensor(components if dimension == 2 else (*components, x[2]))
    interpreter = Interpreter()
    point = {P: np.zeros(dimension)}
    assert interpreter.evaluate(divergence(vector, P), points=point) == pytest.approx(dimension - 2)
    np.testing.assert_allclose(
        interpreter.evaluate(curl(vector, P), points=point), 1 if dimension == 2 else [0, 0, 1]
    )


def test_point_gradient_and_tetrahedron_quadrature():
    interpreter = Interpreter()
    P = EuclideanSpace(0)
    assert interpreter.evaluate(gradient(num(1), P), points={P: []}).shape == (0,)
    K = reference_simplex(3)
    assert interpreter.evaluate(integral(num(1), K)) == pytest.approx(1 / 6)
    assert interpreter.evaluate(integral(coordinate(K)[2], K)) == pytest.approx(1 / 24)


def test_unbounded_euclidean_space_has_no_unit_simplex_quadrature():
    with pytest.raises(NotImplementedError, match="no reference quadrature"):
        Interpreter().evaluate(integral(num(1), EuclideanSpace(2)))


@pytest.mark.parametrize("ambient_dimension", [2, 3])
def test_area_and_mapped_gradient_on_a_thin_triangle(ambient_dimension):
    from uflx.geometry import push_forward, transformation_law
    from uflx_basix.topology import static_topology

    from ..uflx_basix.meshes import vertex_mesh

    topology = static_topology((reference_simplex(2),), ((0, 1, 2),))
    height = 1e-9
    vertices = np.zeros((3, ambient_dimension))
    vertices[1:, 0] = 1
    vertices[2, 1] = height
    mesh, geometry = vertex_mesh(topology, vertices)
    xi = function(space("reference coordinate", domain=topology, scalar_type="real"), "xi")
    oracle = Interpreter(geometry | {xi: lambda p: p[0]})
    np.testing.assert_allclose(oracle.evaluate(integral(num(1), mesh)), height / 2, atol=0)

    # The first reference coordinate is x - y / height on the physical triangle.
    field = push_forward(xi, mesh, transformation_law((), 0))
    point = 0.2 * vertices[1] + 0.3 * vertices[2]
    sample = dict(points={mesh: point})
    assert oracle.evaluate(field, **sample) == pytest.approx(0.2)
    expected = (1, -1 / height, 0)[:ambient_dimension]
    actual = oracle.evaluate(gradient(field, mesh), **sample)
    np.testing.assert_allclose(actual, expected, atol=0)


@pytest.mark.parametrize("cell_name", ["prism", "pyramid"])
def test_basix_vertex_field_reproduces_reference_coordinates(cell_name):
    import basix

    from uflx_basix.basix_cells import basix_cell
    from uflx_basix.topology import static_topology

    cell_type = basix.CellType[cell_name]
    cell = basix_cell(cell_type)
    vertices = tuple(tuple(map(float, vertex)) for vertex in basix.geometry(cell_type))
    topology = static_topology((cell,), (tuple(range(len(vertices))),))
    from ..uflx_basix.meshes import vertex_mesh

    domain, geometry = vertex_mesh(topology, vertices)
    field = domain.coordinate_function
    point = [0.2, 0.1, 0.3]
    np.testing.assert_allclose(
        Interpreter(geometry).evaluate(field, points={topology: point}), point
    )


def test_core_interpretation_without_basix_compiler_or_lowering():
    import subprocess
    import sys

    code = """
import sys

class BlockDependencies:
    def find_spec(self, name, path=None, target=None):
        if name == "basix":
            raise ModuleNotFoundError("optional Basix dependency", name=name)
        if name.startswith(("uflx._src.passes", "uflx_compiler")):
            raise AssertionError(name)

sys.meta_path.insert(0, BlockDependencies())
from uflx.arithmetic import num
from uflx_oracle.interpreter import Interpreter
assert Interpreter().evaluate(num(2) * num(3)) == 6
"""
    subprocess.run([sys.executable, "-c", code], check=True)


def test_basix_form_slot_accepts_an_applied_field():
    import basix

    from uflx_basix.basix_cells import basix_cell
    from uflx_basix.basix_spaces import basix_space

    cell = basix_cell(basix.CellType.triangle)
    V = basix_space("trial", cell, basix.ElementFamily.P, 1)
    u = function(V, "u")
    identity = form(u, (u,))
    field = coordinate(cell)[0]
    point = {cell: [0.2, 0.3]}
    interpreter = Interpreter()
    assert interpreter.evaluate(apply(identity, (field,)), points=point) == pytest.approx(0.2)
    assert interpreter.evaluate(identity, (lambda x: x[1],), points=point) == pytest.approx(0.3)


def test_applied_field_reads_the_cell_where_the_form_samples_it():
    from ..uflx_basix.integrals import GEOMETRY, D

    V = space("piecewise", domain=D, scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    expression = apply(form(integral(v, D), (v,)), (u,))

    # The unit square consists of two triangles of area 1/2.
    def sample(point):
        return interpreter.evaluate(coordinate(D)[0], points={D: point})

    interpreter = Interpreter(GEOMETRY | {u: {0: sample, 1: lambda point: sample(point) + 2}})
    assert interpreter.evaluate(expression) == pytest.approx(1.5)


def test_surface_gradient_uses_the_supplied_slot_on_a_shared_edge():
    from ..uflx_basix.surface import GEOMETRY, M

    # The two triangles meet at this point but have different tangent planes.
    points = {M: [0.5, 0.5, 0.25]}
    expr = gradient(coordinate(M)[2], M)
    oracle = Interpreter(GEOMETRY)
    with pytest.raises(ValueError, match="local cell slot"):
        oracle.evaluate(expr, points=points)
    for slot, expected in enumerate((np.array([0, 2, 1]) / 5, np.array([-2, 4, 5]) / 9)):
        actual = oracle.evaluate(expr, points=points, cells={M: slot})
        np.testing.assert_allclose(actual, expected, atol=1e-14)
