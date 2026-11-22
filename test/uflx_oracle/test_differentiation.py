"""Analytic derivatives of expressions, cell bases and opaque library operations."""

import math

import numpy as np
import pytest

from uflx._src.extensions.external_operators import AbstractOperation, OutputType, external_operator
from uflx.arithmetic import conjugate, num
from uflx.calculus import coordinate, directional_derivative, gradient
from uflx.forms import function
from uflx.nodes import EuclideanSpace
from uflx.tensors import determinant, tensor
from uflx_oracle.interpreter import Interpreter, Pointwise

from ..external_operators import Energy, EnergyTangent
from ..uflx.spaces import space


def test_derivatives_at_scales_where_perturbations_round_away():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    interpreter = Interpreter()
    assert interpreter.evaluate(gradient(x, P), points={P: [1e24]})[0] == 1
    u = function(space("scalar", scalar_type="real"), "u")
    d = directional_derivative(u * u, u, num(1e24), num(2))
    assert interpreter.evaluate(d) == 4e24


def test_nested_spatial_and_directional_seeds():
    P = EuclideanSpace(1)
    u = function(space("field", domain=P, scalar_type="real"), "u")
    x = coordinate(P)[0]
    interpreter = Interpreter({u: lambda p: p[0] ** 3})
    d = directional_derivative(u * u, u, u, x)
    for expr in (gradient(d, P), directional_derivative(gradient(u * u, P), u, u, x)):
        np.testing.assert_allclose(interpreter.evaluate(expr, points={P: [2]}), [64])
    hessian = gradient(gradient(u, P), P)
    np.testing.assert_allclose(interpreter.evaluate(hessian, points={P: [2]}), [[12]])


def test_real_linear_complex_variation_and_determinant_at_singular_matrix():
    u = function(space("complex"), "u")
    d = directional_derivative(conjugate(u) * u, u, num(1 + 2j), num(3 + 4j))
    assert Interpreter().evaluate(d) == 22
    A = tensor((tensor((u, num(0))), tensor((num(0), num(3)))))
    assert Interpreter().evaluate(directional_derivative(determinant(A), u, num(0), num(2))) == 6


class Sine(AbstractOperation):
    identifier = "test.oracle.Sine"

    def output_type(self, operands):
        return OutputType((), "real", operands[0].domain)

    def derivative_action(self):
        return SineAction()


class SineAction(AbstractOperation):
    identifier = "test.oracle.SineAction"

    def output_type(self, operands):
        return OutputType((), "real", operands[0].domain)


def test_pointwise_external_uses_declared_action_for_spatial_and_field_derivatives():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    u = function(space("field", domain=P, scalar_type="real"), "u")
    response = external_operator(Sine(), (u * u,))
    env = {u: lambda p: p[0], Sine(): Pointwise(math.sin)}
    point = {P: [2]}
    assert Interpreter(env).evaluate(response, points=point) == pytest.approx(math.sin(4))
    with pytest.raises(TypeError, match="derivative action"):
        Interpreter(env).evaluate(gradient(response, P), points=point)
    env[SineAction()] = Pointwise(lambda v, dv: math.cos(v) * dv)
    interpreter = Interpreter(env)
    assert interpreter.evaluate(gradient(response, P), points=point)[0] == pytest.approx(
        4 * math.cos(4)
    )
    d = directional_derivative(response, u, x, num(3))
    assert interpreter.evaluate(d, points=point) == pytest.approx(12 * math.cos(4))


def test_whole_field_external_action_receives_fields_and_ordered_directions():
    P = EuclideanSpace(1)
    u = function(space("field", domain=P, scalar_type="real"), "u")
    a = function(space("parameter", scalar_type="real"), "a")
    expr = external_operator(Energy(), (u, a))
    point = np.array([0.25])
    env = {
        u: lambda x: x[0] ** 2,
        a: 2,
        Energy(): lambda field, scale: field(point) * scale,
        EnergyTangent(): lambda field, scale, dfield, dscale: (
            dfield(point) * scale + field(point) * dscale
        ),
    }
    d = directional_derivative(expr, (u, a), (u, a), (coordinate(P)[0], num(3)))
    assert Interpreter(env).evaluate(d) == pytest.approx(0.25 * 2 + 0.25**2 * 3)
    v = function(a.space, "independent")
    del env[EnergyTangent()]
    assert Interpreter(env).evaluate(directional_derivative(expr, v, num(0), num(1))) == 0


def test_opaque_field_can_supply_a_jvp_only_when_needed():
    class Field:
        def __call__(self, point):
            return math.sin(point[0])

        def jvp(self, point, direction):
            return math.cos(point[0]) * direction[0]

    P = EuclideanSpace(1)
    u = function(space("opaque", domain=P, scalar_type="real"), "u")
    assert Interpreter({u: Field()}).evaluate(gradient(u, P), points={P: [0.25]})[
        0
    ] == pytest.approx(math.cos(0.25))
    opaque = lambda x: math.sin(x[0])
    assert Interpreter({u: opaque}).evaluate(u, points={P: [0.25]}) == math.sin(0.25)
    with pytest.raises(TypeError, match=r"jvp|derivative action"):
        Interpreter({u: opaque}).evaluate(gradient(u, P), points={P: [0.25]})


def test_basix_second_derivatives_come_from_basis_tabulation():
    import basix

    from uflx_basix.basix_cells import basix_cell
    from uflx_basix.basix_spaces import basix_space

    K = basix_cell(basix.CellType.triangle)
    V = basix_space("quadratic", K, basix.ElementFamily.P, 2)
    u = function(V, "u")
    x, y = V.element(0).points.T
    interpreter = Interpreter({u: x * x + 3 * x * y + 2 * y * y})
    np.testing.assert_allclose(
        interpreter.evaluate(gradient(gradient(u, K), K), points={K: [0.23, 0.17]}),
        [[2, 3], [3, 4]],
        atol=1e-13,
    )


def test_pointwise_action_is_needed_only_for_an_active_operand():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    law = Sine()
    interpreter = Interpreter({law: Pointwise(math.sin)})
    response = external_operator(law, (x * x,))
    assert interpreter.evaluate(gradient(response, P), points={P: [0]})[0] == 0
    with pytest.raises(TypeError, match="derivative action"):
        interpreter.evaluate(gradient(gradient(response, P), P), points={P: [0]})


def test_second_spatial_derivative_on_a_curved_mesh():
    from uflx.geometry import mesh
    from uflx_basix.cells import reference_simplex
    from uflx_basix.nodes import VertexCoordinates
    from uflx_basix.topology import static_topology

    T = static_topology((reference_simplex(1),), ((0, 1),))
    t = VertexCoordinates(T, ((0.0,), (1.0,)))[0]
    curve = mesh(tensor((t, t * t)), EuclideanSpace(2))
    x = coordinate(curve)[0]
    # grad(x) = (1, 2t) / (1 + 4t**2). Differentiate along the curve at t = 1/2.
    actual = Interpreter().evaluate(
        gradient(gradient(x, curve), curve), points={curve: [0.5, 0.25]}, cells={curve: 0}
    )
    np.testing.assert_allclose(actual, [[-0.5, -0.5], [0, 0]], atol=1e-14)
