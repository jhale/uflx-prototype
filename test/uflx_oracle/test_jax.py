"""Trace the original UFLx graph through the public evaluation entry point."""

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from uflx._src.extensions.external_operators import external_operator
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import function
from uflx_basix.cells import reference_simplex
from uflx_oracle.interpreter import Interpreter, Pointwise

from ..external_operators import Energy
from ..uflx.spaces import space
from .test_differentiation import Sine


def test_jit_value_and_gradient_of_integrated_spatial_derivative():
    cell = reference_simplex(1)
    x = coordinate(cell)[0]
    a = function(space("scale", scalar_type="real"), "a")
    slope = gradient(a * x * x, cell)[0]
    expression = integral(slope * slope, cell)

    def energy(value):
        return Interpreter({a: value}).evaluate(expression)

    values, gradients = jax.jit(jax.vmap(jax.value_and_grad(energy)))(jnp.array([3.0, 1.5]))
    np.testing.assert_allclose(values, [12, 3])
    np.testing.assert_allclose(gradients, [8, 4])


def test_jit_differentiates_whole_field_samples():
    cell = reference_simplex(1)
    a = function(space("scale", scalar_type="real"), "a")
    x = coordinate(cell)[0]
    law = Energy()
    expression = external_operator(law, (a * x, a))

    def read(field, scale):
        return scale * (field(jnp.array([0.25])) + field(jnp.array([0.75])))

    def run(value):
        return Interpreter({a: value, law: read}).evaluate(expression)

    val, deriv = jax.jit(jax.value_and_grad(run))(2.0)
    np.testing.assert_allclose([val, deriv], [4.0, 4.0])


def test_basix_tabulation_at_traced_points_and_coefficients():
    import basix

    from uflx_basix.basix_cells import basix_cell
    from uflx_basix.basix_spaces import basix_space

    cell = basix_cell(basix.CellType.interval)
    V = basix_space("quadratic", cell, basix.ElementFamily.P, 2)
    u = function(V, "u")
    dofs = V.element(0).points[:, 0] ** 2
    slope = gradient(u, cell)[0]

    def sample(dofs, point):
        return Interpreter({u: dofs}).evaluate(slope, points={cell: point})

    f = jax.jit(jax.vmap(jax.value_and_grad(sample, argnums=1), in_axes=(None, 0)))
    values, hessians = f(dofs, jnp.array([[0.375], [0.75]]))
    np.testing.assert_allclose(values, [0.75, 1.5])
    np.testing.assert_allclose(hessians, [[2.0], [2.0]])


def test_batched_basix_mixed_derivatives_on_a_tetrahedron():
    import basix

    from uflx_basix.basix_cells import basix_cell
    from uflx_basix.basix_spaces import basix_space

    cell = basix_cell(basix.CellType.tetrahedron)
    V = basix_space("cubic", cell, basix.ElementFamily.P, 3)
    u = function(V, "u")
    r, s, t = V.element(0).points.T
    oracle = Interpreter({u: r * s * t + r * r * s})
    dr_ds = gradient(gradient(u, cell), cell)[0, 1]

    def sample(point):
        return oracle.evaluate(dr_ds, points={cell: point})

    points = jnp.array([[0.2, 0.3, 0.1], [0.4, 0.1, 0.25]])
    mixed, derivatives = jax.jit(jax.vmap(jax.value_and_grad(sample)))(points)
    np.testing.assert_allclose(mixed, 2 * points[:, 0] + points[:, 2], atol=1e-12)
    np.testing.assert_allclose(derivatives, [[2, 0, 1], [2, 0, 1]], atol=1e-12)


@pytest.mark.parametrize("reference_precision", [False], indirect=True)
def test_external_jax_custom_jvp_transposes_under_jit(reference_precision):
    a = function(space("scale", scalar_type="real"), "a")
    law = Energy()
    expr = external_operator(law, (a, -a))

    @jax.custom_jvp
    def maximum(x, y):
        return jnp.maximum(x, y)

    @maximum.defjvp
    def tangent(primals, directions):
        x, y = primals
        dx, dy = directions
        # Choose the first operand's derivative at a tie.
        return maximum(x, y), jnp.where(x >= y, dx, dy)

    def read(value):
        return Interpreter({a: value, law: maximum}).evaluate(expr)

    assert jax.jit(jax.grad(read))(0.0) == 1.0


def test_spatial_derivative_of_a_field_returned_by_a_library():
    cell = reference_simplex(1)
    a = function(space("scale", scalar_type="real"), "a")
    x = coordinate(cell)[0]
    law = Sine()
    expr = external_operator(law, (a * x,))

    def evaluate(value, point):
        library = {
            a: value,
            law: lambda field: lambda p: p[0] * jnp.sin(field(p)),
        }
        return Interpreter(library).evaluate(expr, points={cell: point})

    value, (da, dx) = jax.jit(jax.value_and_grad(evaluate, argnums=(0, 1)))(2.0, jnp.array([3.0]))
    np.testing.assert_allclose(
        [value, da, dx[0]], [3 * math.sin(6), 9 * math.cos(6), math.sin(6) + 6 * math.cos(6)]
    )


def test_pointwise_jax_binding_is_traced_directly():
    a = function(space("scale", scalar_type="real"), "a")
    law = Sine()
    expr = external_operator(law, (a,))

    def read(value):
        return Interpreter({a: value, law: Pointwise(jnp.sin)}).evaluate(expr)

    np.testing.assert_allclose(jax.jit(jax.grad(read))(0.5), math.cos(0.5))
