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


def vector_type(operands):
    x = operands[0]
    return OutputType(x.value_shape, x.scalar_type, x.domain)


pullback = operation("test.response_pullback", 2, vector_type)


class Response(AbstractOperation):
    identifier = "test.ResponseVJP"
    output_type = staticmethod(vector_type)

    def vjp_action(self, operand):
        assert operand == 0
        return pullback


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
        return jnp.array([x[1] * seed[0] + seed[1], x[0] * seed[0] + 2 * seed[1]])

    R = compile(residual, law, response, u)
    JT = compile(adjoint, pullback, response_vjp, u, z)
    values = np.array([1.0, 2.0, 3.0, 4.0])
    seed = np.array([2.0, -1.0, 1.0, 3.0])
    jac = jax.jacfwd(R)(values)
    assert not np.allclose(jac @ seed, jac.T @ seed)
    np.testing.assert_allclose(JT(values, seed), jac.T @ seed, rtol=1e-6)
    np.testing.assert_allclose(compile(transpose, pullback, response_vjp, u)(values), jac.T)


def test_chain_rule_sums_vjps_for_repeated_inputs_and_holds_seed_fixed():
    u, v, h = (function(real_scalar_space(), n) for n in ("u", "v", "h"))
    left = operation("test.product_left", 3, vector_type)
    right = operation("test.product_right", 3, vector_type)

    class Multiply(AbstractOperation):
        identifier = "test.MultiplyVJP"
        output_type = staticmethod(vector_type)

        def vjp_action(self, operand):
            return (left, right)[operand]

    law = Multiply()
    square = external_operator(law, (u, u))
    cubic = external_operator(law, (square, u))
    residual = form((cubic + u) * v, (v,))
    # z = u is substituted after differentiation: z * (3*u**2 + 1) * h.
    adjoint = vjp(residual, u, u)
    evaluate = trace(adjoint, law, left, right, u)
    result = evaluate(lambda x, y: x * y, lambda x, y, z: y * z, lambda x, y, z: x * z, 2.0, 3.0)
    assert result == 78.0
    both = apply(adjoint, (h,))
    assert (
        trace(both, law, left, right, u, h)(
            lambda x, y: x * y, lambda x, y, z: y * z, lambda x, y, z: x * z, 2.0, 3.0
        )
        == result
    )


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
    push = pairing(forward, z, law, law.derivative_action(), u, h, z)
    pull = pairing(reverse, h, law.vjp_action(0), u, h, z)
    lhs = push(Pointwise(lambda x: A @ x), Pointwise(lambda x, p: A @ p), values, delta, seed)
    rhs = pull(Pointwise(lambda x, p: A.T @ p), values, delta, seed)
    np.testing.assert_allclose(lhs, rhs, atol=1e-6)


def test_partial_linearization_dispatches_only_the_requested_external_rule(monkeypatch):
    V = real_scalar_space()
    u, v = (function(V, name) for name in ("u", "v"))
    law = operation("test.mode_selection", 1, vector_type)
    push, pull = law.derivative_action(), law.vjp_action(0)
    R = form(external_operator(law, (u,)) * v, (v,))

    def unexpected(*args):
        pytest.fail("requested the unused derivative rule")

    with monkeypatch.context() as guard:
        guard.setattr(AbstractOperation, "derivative_action", unexpected)
        guard.setattr(AbstractOperation, "vjp_action", unexpected)
        B = linearize(R, u)
        direction, test = B.arguments
        # The seed depends on u but is held fixed during differentiation.
        seeded = partial_form(B, {test: u})
        pushed = partial_form(B, {direction: u})

    with monkeypatch.context() as guard:
        guard.setattr(AbstractOperation, "derivative_action", unexpected)
        scalar = apply_external_derivatives(apply(seeded, (u,)))
        assert trace(scalar, pull, u)(lambda x, z: 2 * x * z, 3.0) == 54.0

    with monkeypatch.context() as guard:
        guard.setattr(AbstractOperation, "vjp_action", unexpected)
        for pairing in (apply(pushed, (u,)), apply(B, (u, u))):
            scalar = apply_external_derivatives(pairing)
            assert trace(scalar, law, push, u)(lambda x: x * x, lambda x, h: 2 * x * h, 3.0) == 54.0
