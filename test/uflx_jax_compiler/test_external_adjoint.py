# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

"""Assemble a symbolic external VJP on a P1 interval."""

from functools import partial

import basix
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import integral, linearize
from uflx.forms import apply, form, function
from uflx.forms import partial as partial_form
from uflx_basix.cells import cell
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space, real_scalar_space
from uflx_extensions.external_operators import (
    AbstractOperation,
    OutputType,
    apply_external_derivatives,
    external_operator,
    jvp,
    operation,
    vjp,
)
from uflx_jax_compiler import trace
from uflx_jax_compiler.interpreter import Pointwise

from ..uflx.spaces import space


def vector_type(operands):
    x = operands[0]
    return OutputType(x.value_shape, x.scalar_type, x.domain)


class Response(AbstractOperation):
    identifier = "test.ResponseVJP"
    output_type = staticmethod(vector_type)


def test_external_adjoint_assembles_transpose_without_a_tangent_rule():
    K = cell(basix.CellType.interval)
    P1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    V = local_space(K, P1, shape=(2,))
    u, v, z, h = (function(V, label) for label in ("u", "v", "z", "h"))
    law = Response()
    residual = form(integral(external_operator(law, (u,)) @ v, K), (v,))
    adjoint = vjp(residual, u, z)
    (direction,) = adjoint.arguments
    assert direction.space == h.space
    transpose = form(adjoint.expression, (direction, z))

    def compile(f, op, impl, *inputs):
        body = apply_quadrature(f.expression, *basix.make_quadrature(K.cell_type, 3))
        return jax.jit(partial(trace(form(body, f.arguments), op, *inputs), Pointwise(impl)))

    def response(x):
        return jnp.array([x[0] * x[1], x[0] + 2 * x[1]])

    def response_vjp(x, seed):
        return (jnp.array([x[1] * seed[0] + seed[1], x[0] * seed[0] + 2 * seed[1]]),)

    R = compile(residual, law, response, u)
    JT = compile(adjoint, law.vjp_operator(), response_vjp, u, z)
    values = np.array([1.0, 2.0, 3.0, 4.0])
    seed = np.array([2.0, -1.0, 1.0, 3.0])
    jac = jax.jacfwd(R)(values)
    assert not np.allclose(jac @ seed, jac.T @ seed)
    np.testing.assert_allclose(JT(values, seed), jac.T @ seed, rtol=1e-6)
    np.testing.assert_allclose(compile(transpose, law.vjp_operator(), response_vjp, u)(values), jac.T)


def test_chain_rule_sums_vjps_for_repeated_inputs_and_holds_seed_fixed():
    u, v, h = (function(real_scalar_space(), n) for n in ("u", "v", "h"))
    law = operation("test.product", 2, vector_type)

    def multiply(x, y):
        return x * y

    def pullback(x, y, seed):
        return y * seed, x * seed

    square = external_operator(law, (u, u))
    cubic = external_operator(law, (square, u))
    residual = form((cubic + u) * v, (v,))
    # z = u is substituted after differentiation: z * (3*u**2 + 1) * h.
    adjoint = vjp(residual, u, u)
    evaluate = trace(adjoint, law, law.vjp_operator(), u)
    result = evaluate(multiply, pullback, 2.0, 3.0)
    assert result == 78.0
    both = apply(adjoint, (h,))
    assert trace(both, law, law.vjp_operator(), u, h)(multiply, pullback, 2.0, 3.0) == result


def test_jvp_direction_remains_free_when_it_is_also_the_test_function():
    V = real_scalar_space()
    u, v = (function(V, name) for name in ("u", "v"))
    square = operation("test.square_bound_direction", 1, vector_type)
    residual = form(external_operator(square, (u,)) * v, (v,))

    tangent = jvp(residual, u, v)
    (test,) = tangent.arguments
    assert test != v
    kernel = trace(tangent, square, square.jvp_operator(), u, v)
    assert kernel(lambda x: x * x, lambda x, h: 2 * x * h, 2.0, 3.0, 5.0) == 60.0


def test_unsupported_external_component_selection_is_explicit():
    K = cell(basix.CellType.interval)
    P1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    V = local_space(K, P1, shape=(2,))
    u, v, h = (function(V, n) for n in ("u", "v", "h"))
    residual = form(integral(external_operator(Response(), (u,))[0] * v[0], K), (v,))
    with pytest.raises(TypeError, match="unsupported external adjoint through Indexed"):
        vjp(residual, u, h)


def test_public_jvp_vjp_pairing_for_automatically_derived_actions():
    K = cell(basix.CellType.interval)
    P1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    V = local_space(K, P1, shape=(2,))
    u, v, h, z = (function(V, name) for name in ("u", "v", "h", "z"))
    law = operation("test.linear_map", 1, vector_type)
    residual = form(integral(external_operator(law, (u,)) @ v, K), (v,))
    forward, reverse = jvp(residual, u, h), vjp(residual, u, z)
    assert forward.arguments == (v,)
    assert reverse.arguments[0].space == V
    assert reverse.arguments[0] not in (u, v, h, z)

    rule = basix.make_quadrature(K.cell_type, 2)

    def pairing(linear_form, slot_value, *inputs):
        body = apply_quadrature(linear_form.expression, *rule)
        return trace(apply(form(body, linear_form.arguments), (slot_value,)), *inputs)

    A = jnp.array([[2.0, 3.0], [-1.0, 4.0]])
    values = np.array([1.0, 2.0, 3.0, 4.0])
    delta = np.array([2.0, 0.0, 0.0, 1.0])
    seed = np.array([0.0, 1.0, 3.0, 0.0])
    push = pairing(forward, z, law, law.jvp_operator(), u, h, z)
    pull = pairing(reverse, h, law.vjp_operator(), u, h, z)
    lhs = push(Pointwise(lambda x: A @ x), Pointwise(lambda x, p: A @ p), values, delta, seed)
    rhs = pull(Pointwise(lambda x, p: (A.T @ p,)), values, delta, seed)
    np.testing.assert_allclose(lhs, rhs, atol=1e-6)


def test_partial_linearization_dispatches_only_the_requested_external_rule(monkeypatch):
    V = real_scalar_space()
    u, v = (function(V, name) for name in ("u", "v"))
    law = operation("test.mode_selection", 1, vector_type)
    push, pull = law.jvp_operator(), law.vjp_operator()
    R = form(external_operator(law, (u,)) * v, (v,))

    def unexpected(*args):
        pytest.fail("requested the unused derivative rule")

    with monkeypatch.context() as guard:
        guard.setattr(AbstractOperation, "jvp_operator", unexpected)
        guard.setattr(AbstractOperation, "vjp_operator", unexpected)
        B = linearize(R, u)
        direction, test = B.arguments
        # The seed depends on u but is held fixed during differentiation.
        seeded = partial_form(B, {test: u})
        pushed = partial_form(B, {direction: u})

    with monkeypatch.context() as guard:
        guard.setattr(AbstractOperation, "jvp_operator", unexpected)
        scalar = apply_external_derivatives(apply(seeded, (u,)))
        assert trace(scalar, pull, u)(lambda x, z: (2 * x * z,), 3.0) == 54.0

    with monkeypatch.context() as guard:
        guard.setattr(AbstractOperation, "vjp_operator", unexpected)
        for pairing in (apply(pushed, (u,)), apply(B, (u, u))):
            scalar = apply_external_derivatives(pairing)
            assert trace(scalar, law, push, u)(lambda x: x * x, lambda x, h: 2 * x * h, 3.0) == 54.0


def test_scalar_objective_vjp_matches_seeded_residual_and_analytic_gradient():
    K = cell(basix.CellType.interval)
    P1 = basix.create_element(basix.ElementFamily.P, K.cell_type, 1)
    u = function(local_space(K, P1), "u")
    test = function(real_scalar_space(), "test")
    law = operation("test.energy_density", 1, vector_type)
    density = external_operator(law, (u,))
    J = integral(density, K)
    seed = num(2)
    direct = vjp(J, u, seed)
    assert vjp(form(J, ()), u, seed) is direct
    lifted = vjp(form(integral(density * test, K), (test,)), u, seed)
    rule = basix.make_quadrature(K.cell_type, 2)
    for derivative in (direct, lifted):
        sampled = apply_quadrature(derivative.expression, *rule)
        kernel = trace(form(sampled, derivative.arguments), law.vjp_operator(), u)
        # J = integral(u²/2), u has nodal values [1, 4]. The gradient is M u.
        values = kernel(Pointwise(lambda x, bar: (x * bar,)), np.array([1.0, 4.0]))
        np.testing.assert_allclose(values, [2.0, 3.0], rtol=1e-6)


def test_scalar_objective_seed_is_held_fixed():
    u = function(real_scalar_space(), "u")
    square = operation("test.square_objective", 1, vector_type)
    J = external_operator(square, (u,))
    gradient = vjp(J, u, u)
    evaluate = trace(gradient, square.vjp_operator(), u)
    # At u=3 in direction 2: seed * (2*u) * 2.
    assert evaluate(lambda x, bar: (2 * x * bar,), 3.0, 2.0) == 36.0


def test_one_pullback_serves_scalar_and_vector_operands():
    a = function(real_scalar_space(), "a")
    x = function(space("vector", None, (2,), scalar_type="real"), "x")
    law = operation("test.scaled_norm", 2, lambda operands: OutputType((), "real", None))
    objective = external_operator(law, (a, x))
    da = trace(vjp(objective, a, num(1)), law.vjp_operator(), a, x)
    dx = trace(vjp(objective, x, num(1)), law.vjp_operator(), a, x)

    def pullback(scale, vector, seed):
        _, back = jax.vjp(lambda s, v: s * (v @ v), scale, vector)
        return back(seed)

    values = jnp.array([2.0, 3.0])
    assert da(pullback, 4.0, values, 1.0) == 13.0
    assert dx(pullback, 4.0, values, jnp.array([0.0, 1.0])) == 24.0
    with pytest.raises(ValueError, match="tuple of 2 cotangents"):
        da(lambda a, x, seed: (x @ x * seed,), 4.0, values, 1.0)
