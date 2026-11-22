"""Regressions for rewrite completion, extension boundaries and value substitution."""

from __future__ import annotations

import numpy as np
import pytest

from uflx._src.nodes import IndexSum
from uflx.arithmetic import conjugate, num
from uflx.nodes import Complex, Float, Integer, ScalarType, Zero
from uflx.passes import apply_derivatives, lower
from uflx.tensors import component_tensor, index, index_sum, tensor
from uflx_compiler.algebra import _normalizer
from uflx_compiler.evaluation import evaluate
from uflx_compiler.simplification import instantiated, simplify


@pytest.mark.parametrize("rewrite", [lower, _normalizer(), simplify])
@pytest.mark.parametrize(
    "imaginary", [Integer(0), Float(0.0), Float(-0.0), Integer(3), Float(0.25)]
)
def test_complex_branch_arithmetic_survives_rewriting(rewrite, imaginary):
    e = conjugate(Complex(Integer(-1), imaginary)) ** Float(0.5)
    out = rewrite(e)
    assert out.scalar_type == ScalarType.complex
    assert np.isclose(evaluate(out, {}), evaluate(e, {}))
    assert np.isfinite(evaluate(out, {}))


def test_derivative_pass_does_not_implicitly_normalize_algebra():
    e = Integer(2) + Integer(3)
    assert apply_derivatives(e) is e
    assert _normalizer()(e) is Integer(5)


@pytest.mark.parametrize("bind", [index_sum, component_tensor])
def test_simplification_completes_a_child_before_binding_its_index(bind):
    i = index("lost-child-index")
    e = bind(Integer(1) + Zero((), ((i, 2),), (), ScalarType.real), (i,))
    out = simplify(e)
    assert out.free_indices == ()
    assert out.value_shape == e.value_shape
    assert np.allclose(evaluate(out, {}), evaluate(e, {}))
    assert simplify(out) is out


def test_compiler_index_instantiation_avoids_capture():
    i, j = index("outer"), index("inner")
    a, b = tensor((num(2), num(3))), tensor((num(5), num(7)))
    e = index_sum(a[i] * b[j], (j,))
    out = instantiated(e, {i: j})
    assert isinstance(out, IndexSum)
    assert out.indices != (j,)
    assert out.free_indices == ((j, 2),)
    assert np.allclose(evaluate(out, {}), [24, 36])
