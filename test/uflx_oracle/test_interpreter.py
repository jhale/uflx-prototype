"""Analytic checks of the reference interpreter, independent of lowering and compilation."""

from contextvars import Context

import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate, curl, divergence, gradient, integral
from uflx.forms import apply, form, function
from uflx.nodes import EuclideanSpace
from uflx.tensors import index, tensor
from uflx_basix.cells import reference_simplex
from uflx_oracle.interpreter import Interpreter, at, evaluate

from ..uflx.spaces import space


def test_calls_read_current_bindings_and_share_subexpressions_within_one_call():
    V = space("coefficient", scalar_type="real")
    u = function(V, "u")
    calls = 0

    def coefficient():
        nonlocal calls
        calls += 1
        return 3

    env = {u: coefficient}
    interpreter = Interpreter(env)
    expression = u + u
    assert interpreter.evaluate(expression) == 6
    assert calls == 1
    bound = evaluate(expression, env)
    assert bound(()) == 6
    env[u] = 4
    assert interpreter.evaluate(expression) == bound(()) == 8
    assert interpreter.evaluate(form(expression, (u,)), arguments=(5,)) == 10
    with pytest.raises(ValueError, match="expected 1 form argument"):
        interpreter.evaluate(form(expression, (u,)))


def test_cell_point_and_free_index_are_explicit_inputs():
    from uflx_basix.spaces import space

    from ..uflx_basix.integrals import SQUARE

    V = space("vector", (2,), domain=SQUARE)
    u = function(V, "field")
    i = index("component")
    interpreter = Interpreter({u: {0: lambda x: -x, 1: lambda x: 2 * x}})
    for slot, expected in enumerate((-0.3, 0.6)):
        assert interpreter.evaluate(
            u[i], indices={i: 1}, points={SQUARE: [0.2, 0.3]}, cells={SQUARE: slot}
        ) == pytest.approx(expected)


def test_nested_evaluations_restore_points_even_on_failure():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]

    def check():
        assert evaluate(x, {})(()) == 2
        assert Interpreter().evaluate(x, points={P: [3]}) == 3
        with pytest.raises(KeyError):
            Context().run(lambda: evaluate(x, {})(()))
        with pytest.raises(ZeroDivisionError):
            at(P, [4], lambda: 1 / 0)
        return evaluate(x, {})(())

    assert at(P, [2], check) == 2
    with pytest.raises(KeyError):
        evaluate(x, {})(())


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


@pytest.mark.parametrize("cell_name", ["prism", "pyramid"])
def test_basix_vertex_field_reproduces_reference_coordinates(cell_name):
    import basix

    from uflx_basix.basix_cells import basix_cell
    from uflx_basix.nodes import VertexCoordinates
    from uflx_basix.topology import static_topology

    cell_type = basix.CellType[cell_name]
    cell = basix_cell(cell_type)
    vertices = tuple(tuple(map(float, vertex)) for vertex in basix.geometry(cell_type))
    topology = static_topology((cell,), (tuple(range(len(vertices))),))
    field = VertexCoordinates(topology, vertices)
    point = [0.2, 0.1, 0.3]
    np.testing.assert_allclose(Interpreter().evaluate(field, points={topology: point}), point)


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
    from ..uflx_basix.integrals import D

    V = space("piecewise", domain=D, scalar_type="real")
    u, v = function(V, "u"), function(V, "v")
    expression = apply(form(integral(v, D), (v,)), (u,))

    # The unit square consists of two triangles of area 1/2.
    def sample(point):
        return evaluate(coordinate(D)[0], {})(())

    interpreter = Interpreter({u: {0: sample, 1: lambda point: sample(point) + 2}})
    assert interpreter.evaluate(expression) == pytest.approx(1.5)
