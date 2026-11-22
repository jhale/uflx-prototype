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


def test_jit_samples_whole_field_and_honours_declared_action():
    cell = reference_simplex(1)
    a = function(space("scale", scalar_type="real"), "a")
    x = coordinate(cell)[0]
    law = Energy()
    expression = external_operator(law, (a * x, a))

    def read(field, scale):
        return scale * (field(jnp.array([0.25])) + field(jnp.array([0.75])))

    def declared(field, scale, dfield, dscale):
        # A distinct result detects accidental AD through the primal binding.
        return 3 * (scale * dfield(jnp.array([1.0])) + dscale * field(jnp.array([1.0])))

    def run(value, library):
        return Interpreter({a: value, law: read, **library}).evaluate(expression)

    for library, expected in [({}, 4.0), ({law.derivative_action(): declared}, 12.0)]:
        val, deriv = jax.jit(jax.value_and_grad(lambda v: run(v, library)))(2.0)
        np.testing.assert_allclose([val, deriv], [4.0, expected])


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


@pytest.mark.parametrize("reference_precision", [False], indirect=True)
def test_opaque_pointwise_declared_action_transposes_under_jit(reference_precision):
    a = function(space("scale", scalar_type="real"), "a")
    law = Energy()
    expr = external_operator(law, (a, a))
    bindings = {
        law: Pointwise(lambda x, y: math.sin(x) * y),
        law.derivative_action(): Pointwise(
            lambda x, y, h, k: math.cos(x) * y * h + math.sin(x) * k
        ),
    }

    def read(value):
        return Interpreter({a: value, **bindings}).evaluate(expr)

    actual = jax.jit(jax.grad(read))(0.5)
    np.testing.assert_allclose(actual, 0.5 * math.cos(0.5) + math.sin(0.5), rtol=1e-6)


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
            law.derivative_action(): lambda field, dfield: (
                lambda p: p[0] * jnp.cos(field(p)) * dfield(p)
            ),
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
        return Interpreter({a: value, law: Pointwise(jnp.sin, traceable=True)}).evaluate(expr)

    np.testing.assert_allclose(jax.jit(jax.grad(read))(0.5), math.cos(0.5))
