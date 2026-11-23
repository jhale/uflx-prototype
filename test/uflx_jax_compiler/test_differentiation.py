"""Analytic derivatives of expressions, cell bases and external operations."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from uflx.arithmetic import conditional, conjugate, less_than, num
from uflx.calculus import coordinate, directional_derivative, gradient
from uflx.forms import function
from uflx.nodes import EuclideanSpace
from uflx.tensors import determinant, tensor
from uflx_extensions.external_operators import AbstractOperation, OutputType, external_operator
from uflx_jax_compiler.interpreter import Interpreter, Pointwise

from ..uflx.spaces import space as field_space
from ..uflx_extensions.operations import Energy


def test_derivatives_at_scales_where_perturbations_round_away():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    interpreter = Interpreter()
    assert interpreter.evaluate(gradient(x, P), points={P: [1e24]})[0] == 1
    u = function(field_space("scalar", scalar_type="real"), "u")
    d = directional_derivative(u * u, u, num(1e24), num(2))
    assert interpreter.evaluate(d) == 4e24
    for exponent in (num(0), num(0.0), num(1)):
        derivative = directional_derivative(u**exponent, u, num(0), num(1))
        assert interpreter.evaluate(derivative) == exponent.value


def test_nested_spatial_and_directional_seeds():
    P = EuclideanSpace(1)
    u = function(field_space("field", domain=P, scalar_type="real"), "u")
    x = coordinate(P)[0]
    interpreter = Interpreter({u: x ** num(3)})
    d = directional_derivative(u * u, u, u, x)
    for expr in (gradient(d, P), directional_derivative(gradient(u * u, P), u, u, x)):
        np.testing.assert_allclose(interpreter.evaluate(expr, points={P: [2]}), [64])
    hessian = gradient(gradient(u, P), P)
    np.testing.assert_allclose(interpreter.evaluate(hessian, points={P: [2]}), [[12]])


def test_real_linear_complex_variation_and_determinant_at_singular_matrix():
    u = function(field_space("complex"), "u")
    d = directional_derivative(conjugate(u) * u, u, num(1 + 2j), num(3 + 4j))
    assert Interpreter().evaluate(d) == 22
    A = tensor((tensor((u, num(0))), tensor((num(0), num(3)))))
    assert Interpreter().evaluate(directional_derivative(determinant(A), u, num(0), num(2))) == 6


class Sine(AbstractOperation):
    identifier = "test.oracle.Sine"

    def output_type(self, operands):
        return OutputType((), "real", operands[0].domain)


def test_spatial_and_field_derivatives_of_pointwise_external():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    u = function(field_space("field", domain=P, scalar_type="real"), "u")
    response = external_operator(Sine(), (u * u,))
    interpreter = Interpreter({u: x, Sine(): Pointwise(jnp.sin)})
    point = {P: [2]}
    assert interpreter.evaluate(response, points=point) == pytest.approx(math.sin(4))
    assert interpreter.evaluate(gradient(response, P), points=point)[0] == pytest.approx(
        4 * math.cos(4)
    )
    d = directional_derivative(response, u, x, num(3))
    assert interpreter.evaluate(d, points=point) == pytest.approx(12 * math.cos(4))


def test_second_derivative_of_pointwise_external():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    response = external_operator(Sine(), (x * x,))
    hessian = gradient(gradient(response, P), P)
    oracle = Interpreter({Sine(): Pointwise(jnp.sin)})
    for x0 in (0.0, 2.0):
        result = oracle.evaluate(hessian, points={P: [x0]})
        np.testing.assert_allclose(result, [[2 * math.cos(x0**2) - 4 * x0**2 * math.sin(x0**2)]])


def test_directional_derivative_of_whole_field_external():
    P = EuclideanSpace(1)
    u = function(field_space("field", domain=P, scalar_type="real"), "u")
    a = function(field_space("parameter", scalar_type="real"), "a")
    expr = external_operator(Energy(), (u, a))
    point = np.array([0.25])
    env = {
        u: lambda x: x[0] ** 2,
        a: 2,
        Energy(): lambda field, scale: field(point) * scale,
    }
    d = directional_derivative(expr, (u, a), (u, a), (coordinate(P)[0], num(3)))
    assert Interpreter(env).evaluate(d) == pytest.approx(0.25 * 2 + 0.25**2 * 3)
    v = function(a.space, "independent")
    assert Interpreter(env).evaluate(directional_derivative(expr, v, num(0), num(1))) == 0


def test_traceable_field_supports_spatial_and_directional_derivatives():
    P = EuclideanSpace(1)
    u = function(field_space("field", domain=P, scalar_type="real"), "u")
    oracle = Interpreter({u: lambda point: point[0] ** 2})
    assert oracle.evaluate(u, points={P: [3]}) == 9
    np.testing.assert_allclose(oracle.evaluate(gradient(u, P), points={P: [3]}), [6])
    derivative = directional_derivative(u * u, u, u, num(2))
    assert oracle.evaluate(derivative, points={P: [3]}) == 36


def test_predicate_can_read_an_opaque_value_while_differentiating_a_branch():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    u = function(field_space("predicate", domain=P, scalar_type="real"), "u")
    field = conditional(less_than(u, num(0)), x * x, x)
    oracle = Interpreter({u: lambda point: float(point[0])})
    assert oracle.evaluate(gradient(field, P), points={P: [-2]})[0] == -4

    response = external_operator(Sine(), (u,))
    field = conditional(less_than(response, num(0)), u * u, u)
    derivative = directional_derivative(field, u, x, num(3))
    # JAX cannot apply forward AD to a custom_vjp function. The comparison
    # only uses its value, even when its operand depends on the variable.
    sine = jax.custom_vjp(jnp.sin)
    sine.defvjp(lambda x: (jnp.sin(x), x), lambda x, g: (jnp.cos(x) * g,))
    oracle = Interpreter({Sine(): Pointwise(sine)})
    assert oracle.evaluate(derivative, points={P: [-2]}) == -12


def test_comparison_evaluates_the_spatial_derivative_in_its_operand():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    slope = gradient(x * x, P)[0]
    field = conditional(less_than(slope, num(0)), x * x, x)
    oracle = Interpreter()

    def sample(point):
        return oracle.evaluate(field, points={P: point})

    assert sample(jnp.array([-1.0])) == 1
    values, derivatives = jax.jit(jax.vmap(jax.value_and_grad(sample)))(jnp.array([[-1.0], [2.0]]))
    np.testing.assert_allclose(values, [1, 2])
    np.testing.assert_allclose(derivatives, [[-2], [1]])


def test_basix_second_derivatives_come_from_basis_tabulation():
    import basix

    from uflx_basix.cells import cell
    from uflx_basix.spaces import space

    K = cell(basix.CellType.triangle)
    V = space("quadratic", basix.create_element(basix.ElementFamily.P, K.cell_type, 2))
    u = function(V, "u")
    x, y = V.element(0).points.T
    interpreter = Interpreter({u: x * x + 3 * x * y + 2 * y * y})
    np.testing.assert_allclose(
        interpreter.evaluate(gradient(gradient(u, K), K), points={K: [0.23, 0.17]}),
        [[2, 3], [3, 4]],
        atol=1e-13,
    )


def test_second_spatial_derivative_on_a_curved_mesh():
    from uflx.geometry import mesh

    from ..uflx.library import static_topology
    from ..uflx_basix.cells import reference_simplex

    T = static_topology((reference_simplex(1),), ((0, 1),))
    t = function(field_space("parameter", domain=T, scalar_type="real"), "t")
    curve = mesh(tensor((t, t * t)), EuclideanSpace(2))
    x = coordinate(curve)[0]
    # grad(x) = (1, 2t) / (1 + 4t**2). Differentiate along the curve at t = 1/2.
    actual = Interpreter({t: lambda p: p[0]}).evaluate(
        gradient(gradient(x, curve), curve), points={curve: [0.5, 0.25]}, cells={curve: 0}
    )
    np.testing.assert_allclose(actual, [[-0.5, -0.5], [0, 0]], atol=1e-14)
