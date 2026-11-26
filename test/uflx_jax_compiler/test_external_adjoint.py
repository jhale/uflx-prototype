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

from uflx.calculus import integral
from uflx.forms import apply, form, function
from uflx.forms import partial as partial_form
from uflx_basix.cells import cell
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space, real_scalar_space
from uflx_extensions.external_operators import (
    AbstractOperation,
    OutputType,
    external_adjoint,
    external_operator,
    operation,
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
    transpose = external_adjoint(residual, u, h)
    assert transpose.arguments == (h, v)
    adjoint = partial_form(transpose, {v: z})
    assert adjoint.arguments == (h,)

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
    transpose = external_adjoint(residual, u, h)
    adjoint = partial_form(transpose, {v: u})
    evaluate = trace(adjoint, law, left, right, u)
    result = evaluate(lambda x, y: x * y, lambda x, y, z: y * z, lambda x, y, z: x * z, 2.0, 3.0)
    assert result == 78.0
    both = apply(transpose, (h, u))
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
        external_adjoint(residual, u, h)
