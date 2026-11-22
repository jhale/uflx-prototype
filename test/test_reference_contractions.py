"""Reference integration must preserve index binding and runtime dependencies."""

import numpy as np
import pytest

from uflx._src.nodes import IndexSum
from uflx._src.nodes.structure import expression_operands
from uflx.arithmetic import num
from uflx.tensors import index, index_sum
from uflx_basix.runtime import runtime_array
from uflx_compiler.c_code import c_kernel
from uflx_compiler.contractions import _preintegrate
from uflx_compiler.evaluation import evaluate
from uflx_compiler.quadrature import constant


def reductions(expression):
    return int(isinstance(expression, IndexSum)) + sum(
        reductions(c) for c in expression_operands(expression)
    )


def example(varying):
    q, i, j = (index(name) for name in ("q", "i", "j"))
    rng = np.random.default_rng(719)
    left = constant(rng.normal(size=(2, 5, 3)))
    right = constant(rng.normal(size=(2, 5, 4)))
    weights = constant(rng.uniform(0.1, 1, size=5))
    material = runtime_array("material", (5, 2, 2) if varying else (2, 2))
    a, b, c, d = (
        material[q, r, s] if varying else material[r, s]
        for r, s in ((0, 0), (0, 1), (1, 0), (1, 1))
    )
    expression = index_sum(
        weights[q]
        * (a * left[0, q, i] + b * left[1, q, i])
        * (c * right[0, q, j] + d * right[1, q, j]),
        (q,),
    )
    return expression, (i, j), rng.normal(size=material.value_shape)


@pytest.mark.parametrize("varying", [False, True])
@pytest.mark.parametrize("share_tables", [False, True])
def test_reference_contractions_match_rectangular_runtime_evaluation(varying, share_tables):
    expression, outputs, data = example(varying)
    result = _preintegrate(expression, outputs, share_tables=share_tables)
    assert result.free_indices == expression.free_indices
    assert reductions(result) == (1 if varying else 0)
    code = c_kernel(expression, "matrix", lambda label, off: f"w[{off}]", output_indices=outputs)
    original = c_kernel(
        expression,
        "matrix",
        lambda label, off: f"w[{off}]",
        output_indices=outputs,
        preintegrate=False,
    )
    assert "for (int q" in original
    assert ("for (int q" in code) == varying
    np.testing.assert_allclose(
        evaluate(result, {"material": data}),
        evaluate(expression, {"material": data}),
        rtol=1e-12,
        atol=1e-12,
    )


@pytest.mark.parametrize("budget", [{"max_terms": 1}, {"max_entries": 1}])
def test_reference_contraction_budget_preserves_original_expression(budget):
    expression, outputs, _ = example(False)
    assert _preintegrate(expression, outputs, **budget) == expression


def test_runtime_quadrature_denominator_is_not_preintegrated():
    expression, outputs, _ = example(False)
    q = expression.indices[0]
    coefficient = runtime_array("coefficient", (5,))
    varying = index_sum(expression.expression / coefficient[q], (q,))
    assert _preintegrate(varying, outputs) == varying


def test_reference_integration_preserves_missing_summand_indices():
    q, i, j = (index(name) for name in ("q", "i", "j"))
    a = runtime_array("a", ())
    left, right = constant(np.arange(3.0)), constant(np.arange(4.0))
    expression = index_sum(a * left[i] * right[j] + constant(np.ones(5))[q] * num(2), (q,))
    result = _preintegrate(expression, (i, j))
    np.testing.assert_allclose(evaluate(result, {"a": 3.0}), evaluate(expression, {"a": 3.0}))
    assert reductions(result) == 0


def test_scalar_scales_share_one_reference_table():
    q, i, j = (index(name) for name in ("q", "i", "j"))
    left = constant(np.arange(1.0, 7.0).reshape(2, 3))
    right = constant(np.arange(1.0, 9.0).reshape(2, 4))
    a = runtime_array("a", (2,))
    expression = index_sum(
        num(2) * a[0] * left[q, i] * right[q, j] + num(3) * a[1] * left[q, i] * right[q, j],
        (q,),
    )
    result = _preintegrate(expression, (i, j), share_tables=True)
    data = np.array([1.25, -0.75])
    np.testing.assert_allclose(evaluate(result, {"a": data}), evaluate(expression, {"a": data}))
    code = c_kernel(result, "scaled", lambda label, off: f"w[{off}]", output_indices=(i, j))
    assert code.count("static const double") == 1


@pytest.mark.parametrize("varying", [False, True])
def test_action_preintegrates_coefficient_dofs_only_with_invariant_geometry(varying):
    q, i, j = (index(name) for name in ("q", "i", "j"))
    rng = np.random.default_rng(12)
    basis = constant(rng.normal(size=(5, 4)))
    coefficients = runtime_array("u", (4,))
    geometry = runtime_array("g", (5,) if varying else ())
    weight = geometry[q] if varying else geometry
    value = index_sum(basis[q, j] * coefficients[j], (j,))
    expression = index_sum(weight * basis[q, i] * value, (q,))
    result = _preintegrate(expression, (i,))
    data = {"u": rng.normal(size=4), "g": rng.normal(size=geometry.value_shape)}
    np.testing.assert_allclose(evaluate(result, data), evaluate(expression, data), atol=1e-12)
    assert (reductions(result) > 0) == varying
    assert _preintegrate(expression, (i,), max_terms=3) == expression


def test_reference_tables_with_equal_coefficients_are_combined():
    """Symmetric cross terms share one precomputed table after coefficient collection."""
    q, i, j = (index(name) for name in ("q", "i", "j"))
    rng = np.random.default_rng(842)
    left = constant(rng.normal(size=(4, 9)))
    right = constant(rng.normal(size=(4, 9)))
    a, b = runtime_array("a", ()), runtime_array("b", ())
    expression = index_sum(
        (a + b) * left[q, i] * right[q, j] + (b + a) * right[q, i] * left[q, j],
        (q,),
    )
    result = _preintegrate(expression, (i, j), share_tables=True)
    assert reductions(result) == 0
    code = c_kernel(
        result,
        "matrix",
        lambda label, off: f"{label}[{off}]",
        output_indices=(i, j),
        preintegrate=False,
    )
    assert code.count("static const double") == 1
    np.testing.assert_allclose(
        evaluate(result, {"a": np.array(1.7), "b": np.array(-0.2)}),
        evaluate(expression, {"a": np.array(1.7), "b": np.array(-0.2)}),
        rtol=1e-12,
        atol=1e-12,
    )
