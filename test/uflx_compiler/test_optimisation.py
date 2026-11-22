"""Compare indexed extraction against the compiler's independent array evaluator."""

import pytest

pytest.importorskip("egglog")

import numpy as np

from uflx._src.nodes import IndexSum, Zero
from uflx._src.optimisation import OptimisationOptions, optimise
from uflx._src.tensors import identity
from uflx.arithmetic import num
from uflx.tensors import component_tensor, index, index_sum, tensor
from uflx_basix.nodes import RuntimeArray
from uflx_compiler.evaluation import evaluate

from ..uflx.test_optimisation import nodes
from ..uflx_basix.test_optimisation import optimise_core


def assert_values(original, result, env):
    assert original.value_shape == result.value_shape
    assert dict(original.free_indices) == dict(result.free_indices)
    assert original.scalar_type == result.scalar_type
    assert original.argument_spaces == result.argument_spaces
    np.testing.assert_allclose(
        evaluate(original, env), evaluate(result, env), rtol=1e-12, atol=1e-12
    )


@pytest.mark.parametrize("reverse", [False, True])
def test_delta_contraction(reverse):
    i, j = index("i"), index("j")
    a = tensor(tuple(num(k + 1) for k in range(4)))
    delta = identity(4)[j, i] if reverse else identity(4)[i, j]
    original = component_tensor(index_sum(delta * a[j], (j,)), (i,))
    result, stats = optimise_core(original, iterations=5)
    assert not any(isinstance(n, IndexSum) for n in nodes(result))
    assert any(region.changed for region in stats.regions)
    assert_values(original, result, {})


def test_diagonal_delta_keeps_the_sum():
    j = index("j")
    a = RuntimeArray("a", (4,))
    original = index_sum(identity(4)[j, j] * a[j], (j,))
    result, _ = optimise_core(original, iterations=5)
    assert any(isinstance(n, IndexSum) for n in nodes(result))
    assert_values(original, result, {"a": np.arange(4.0)})


def test_index_dependent_vector_is_not_treated_as_constant():
    i, j = index("i"), index("j")
    a = RuntimeArray("a", (3,))
    vector = tensor((a[j], a[j] * num(2), a[j] * num(3)))
    original = component_tensor(index_sum(identity(3)[i, j] * vector[j], (j,)), (i,))
    result, _ = optimise_core(original, iterations=5)
    assert_values(original, result, {"a": np.array([1.0, 7.0, 11.0])})


@pytest.mark.parametrize("scalar_type", ["real", "complex"])
def test_zero_under_sum_retains_free_extent_and_type(scalar_type):
    i, j = index("i"), index("j")
    a = tensor(tuple(num(k + 1) for k in range(3)))
    zero = Zero((), ((i, 2), (j, 3)), (), scalar_type)
    original = component_tensor(index_sum(zero * a[j], (j,)), (i,))
    result, _ = optimise_core(original, iterations=5)
    assert_values(original, result, {})
    assert any(isinstance(n, Zero) and n.scalar_type == scalar_type for n in nodes(result))


def test_zero_tensor_packaging():
    original = tensor((tensor((num(0), num(0))), tensor((num(0), num(0)))))
    result, _ = optimise_core(original)
    assert isinstance(result, Zero)
    assert result.value_shape == (2, 2)
    assert_values(original, result, {})


def test_factoring_inside_an_index_sum():
    i = index("i")
    a, b, c = (RuntimeArray(s, (4,)) for s in ("a", "b", "c"))
    original = index_sum(a[i] * b[i] + a[i] * c[i], (i,))
    result, stats = optimise_core(original, iterations=5)
    assert any(region.changed for region in stats.regions)
    assert_values(
        original, result, dict(zip(("a", "b", "c"), np.arange(12.0).reshape(3, 4), strict=True))
    )


def test_shadowed_index_binders():
    i = index("i")
    a = RuntimeArray("a", (3,))
    inner = index_sum(a[i] + a[i], (i,))
    original = index_sum(a[i] * inner, (i,))
    result, _ = optimise_core(original, iterations=5)
    assert_values(original, result, {"a": np.array([2.0, 3.0, 7.0])})


def test_loop_cost_moves_invariant_multiply_outside_sum():
    i = index("i")
    a, c = RuntimeArray("a", (8,)), RuntimeArray("c", ())
    original = index_sum(a[i] * c, (i,))
    result, _ = optimise_core(original, iterations=5, loop_aware=True)
    from uflx._src.nodes import Product

    assert isinstance(result, Product)
    assert isinstance(result.first, IndexSum) or isinstance(result.second, IndexSum)
    assert_values(original, result, {"a": np.arange(8.0), "c": np.array(3.0)})


def test_index_dependent_opaque_factor_stays_inside_sum():
    i = index("i")
    a, b = RuntimeArray("a", (4,)), RuntimeArray("b", (4,))
    original = index_sum(a[i] * (num(1) / b[i]), (i,))
    result, _ = optimise_core(original, iterations=5, loop_aware=True)
    assert isinstance(result, IndexSum)
    assert_values(original, result, {"a": np.arange(4.0), "b": np.arange(1.0, 5.0)})


def test_scalar_region_with_no_independence_facts():
    original = num(2) * num(3) + num(4)
    result, _ = optimise_core(original)
    assert_values(original, result, {})


def test_constant_sum_keeps_its_multiplicity_after_zero_removal():
    i = index("i")
    original = index_sum(Zero((), ((i, 5),), (), "real") + num(3), (i,))
    result, _ = optimise_core(original)
    assert_values(original, result, {})
    assert evaluate(result, {}) == 15


@pytest.mark.parametrize("index_count", [1, 2])
def test_alpha_interning_collects_renamed_opaque_sums(index_count):
    from uflx._src.algorithms.binding import alpha_equivalent

    indices = tuple(index(f"i{k}") for k in range(index_count))
    renamed = tuple(index(f"j{k}") for k in range(index_count))
    a = RuntimeArray("alpha-a", (3,) * index_count)
    left, right = index_sum(a[indices], indices), index_sum(a[renamed], renamed)
    assert left is not right
    assert alpha_equivalent(left, right)
    original = left + right
    # Single-index sums can be opaque by policy; multi-index sums are opaque by default.
    options = OptimisationOptions(
        rule_groups=("arithmetic",) if index_count == 1 else ("arithmetic", "indexed"),
        iterations=3,
    )
    result = optimise(original, options)
    assert sum(r.alpha_merges for r in result.report.regions) >= 1
    assert_values(
        original,
        result.expression,
        {"alpha-a": np.arange(3**index_count).reshape((3,) * index_count)},
    )
    assert sum(isinstance(node, IndexSum) for node in nodes(result.expression)) == 1


def test_options_and_search_reports():
    from uflx._src.optimisation import OptimisationOptions

    with pytest.raises(ValueError, match="iterations"):
        OptimisationOptions(iterations=0)
    with pytest.raises(ValueError, match="rule groups"):
        OptimisationOptions(rule_groups=("unknown",))
    original = num(2) + num(3)
    result = optimise(original)
    assert_values(original, result.expression, {})
    assert result.report.regions
    assert all(r.search_stop in ("saturated", "iteration_limit") for r in result.report.regions)
    assert result.report.total_seconds >= 0


def test_custom_cost_callback():
    from uflx._src.optimisation.costs import loop_cost

    calls = []

    def cost(egraph, expr, children_costs):
        calls.append(1)
        return loop_cost(egraph, expr, children_costs)

    original = num(2) + num(3)
    result = optimise(original, cost_model=cost)
    assert calls
    assert_values(original, result.expression, {})


@pytest.mark.parametrize("kind", ["shared_factors", "cancellation", "indexed_energy"])
def test_derivative_redundancies_surviving_current_cleanup(kind):
    from uflx.calculus import directional_derivative
    from uflx.forms import function
    from uflx.passes import apply_derivatives, lower
    from uflx_basix.spaces import space
    from uflx_compiler.algebra import _normalizer
    from uflx_oracle import evaluate as oracle

    V = space("optimisation-derivative", scalar_type="real")
    u, v, w, h = (function(V, name) for name in ("u", "v", "w", "h"))
    source = u * u * v + u * u * w
    if kind == "cancellation":
        source = (u + v) * (u + v) - u * u - num(2) * u * v - v * v
    elif kind == "indexed_energy":
        X = space("optimisation-vector", value_shape=(4,), scalar_type="real")
        u, h = function(X, "u"), function(X, "h")
        i = index("i")
        source = index_sum(u[i] * u[i] * v + u[i] * u[i] * w, (i,))
    derivative = lower(apply_derivatives(directional_derivative(source, u, u, h)))
    cleaned = _normalizer()(derivative)
    result = optimise(cleaned, OptimisationOptions(iterations=4, max_region_nodes=100)).expression
    from uflx._src.nodes import Product, Sum

    arithmetic = lambda e: sum(isinstance(n, Sum | Product) for n in nodes(e))
    assert arithmetic(result) < arithmetic(cleaned)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        env = {x: rng.uniform(-0.8, 0.8, size=x.value_shape) for x in (u, v, w, h)}
        expected = oracle(derivative, env)(())
        np.testing.assert_allclose(oracle(result, env)(()), expected, rtol=1e-10, atol=1e-10)


def test_coefficient_cancellation_removes_defined_input():
    a = RuntimeArray("cancel", ())
    original = num(2) * a + num(-2) * a
    result = optimise(original, OptimisationOptions(iterations=4)).expression
    assert isinstance(result, Zero)
    assert a not in nodes(result)
    assert_values(original, result, {"cancel": np.array(7.0)})


@pytest.mark.parametrize("groups", [("arithmetic",), ("indexed",), ("arithmetic", "indexed")])
@pytest.mark.parametrize("reverse", [False, True])
def test_zero_product_does_not_erase_partial_expression(groups, reverse):
    x = RuntimeArray("denominator", ())
    partial = num(1) / x
    original = partial * num(0) if reverse else num(0) * partial
    result = optimise(original, OptimisationOptions(rule_groups=groups)).expression
    assert partial in nodes(result)
    with np.errstate(divide="ignore", invalid="ignore"):
        assert np.isnan(evaluate(result, {"denominator": np.array(0.0)}))
    assert_values(original, result, {"denominator": np.array(2.0)})


def test_delta_contracts_defined_runtime_vector_components():
    i, j = index("i"), index("j")
    a = RuntimeArray("unknown-vector", (3,))
    original = component_tensor(index_sum(identity(3)[i, j] * a[j], (j,)), (i,))
    result = optimise(original).expression
    assert not any(isinstance(n, IndexSum) for n in nodes(result))
    assert_values(original, result, {"unknown-vector": np.arange(3.0)})


@pytest.mark.parametrize("groups", [("arithmetic",), ("indexed",), ("arithmetic", "indexed")])
def test_preserving_mode_keeps_grouping_and_finishing_cleanup_disabled(groups):
    x = RuntimeArray("ordered", ())
    original = ((x + num(1.0)) + num(-1.0)) * num(0)
    options = OptimisationOptions(preserve_floating_point_evaluation=True, rule_groups=groups)
    result = optimise(original, options)
    assert result.expression is original
    assert not result.report.regions
    env = {"ordered": np.array(2.0**-54)}
    assert evaluate(original, env) == evaluate(result.expression, env)
    constants = num(2) + num(3)
    assert optimise(constants, options).expression is constants


def test_preserving_mode_keeps_signed_zero_and_reduction_order():
    i = index("ordered-sum")
    a = RuntimeArray("ordered-array", (3,))
    original = index_sum(a[i] * num(2), (i,))
    options = OptimisationOptions(preserve_floating_point_evaluation=True)
    assert optimise(original, options).expression is original
    negative_zero = num(-0.0) + num(0)
    assert optimise(negative_zero, options).expression is negative_zero
    assert not np.signbit(evaluate(negative_zero, {}))


@pytest.mark.parametrize("preserve_floating_point_evaluation", [False, True])
def test_structural_tensor_selection_and_broadcast(preserve_floating_point_evaluation):
    from uflx.tensors import broadcast_to

    options = OptimisationOptions(
        preserve_floating_point_evaluation=preserve_floating_point_evaluation
    )
    original = tensor((tensor((num(2), num(3))), tensor((num(5), num(7)))))[1, 0]
    assert optimise(original, options).expression is num(5)
    i = index("broadcast")
    x = RuntimeArray("broadcast-value", ())
    original = broadcast_to(x, (3,))[i]
    result = optimise(original, options).expression
    assert dict(result.free_indices) == {i: 3}
    assert_values(original, result, {"broadcast-value": np.array(4.0)})
    assert optimise(broadcast_to(x, (3,))[1], options).expression is x


def test_fp_policy_covers_overflow_in_known_defined_arithmetic():
    original = num(0) * (num(1) / num(5e-324))
    relaxed = optimise(original).expression
    preserved = optimise(
        original, OptimisationOptions(preserve_floating_point_evaluation=True)
    ).expression
    assert isinstance(relaxed, Zero)
    assert preserved is original
    with np.errstate(over="ignore", invalid="ignore"):
        assert np.isnan(evaluate(preserved, {}))
    assert evaluate(relaxed, {}) == 0


def test_defined_zero_product_retains_family_indices_without_rewrite_cycle():
    i, j = index("i"), index("j")
    a = RuntimeArray("family-array", (3,))
    original = component_tensor(
        index_sum(Zero((), ((i, 2), (j, 3)), (), "real") * a[j], (j,)), (i,)
    )
    result = optimise(original).expression
    assert a not in nodes(result)
    assert_values(original, result, {"family-array": np.arange(3.0)})


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("operation", ["conjugate", "real_part", "imaginary_part"])
def test_complex_operators_roundtrip_through_egglog(operation, preserve):
    from egglog.deconstruct import get_callable_args

    from uflx._src.optimisation.encoding import decode, encode
    from uflx._src.optimisation.language import Scalar
    from uflx.arithmetic import conjugate, imaginary_part, real_part

    z = num(complex(-0.0, 2.0))
    constructor = {
        "conjugate": conjugate,
        "real_part": real_part,
        "imaginary_part": imaginary_part,
    }[operation]
    original = constructor(z)
    region = encode(original, encode_indices=False, preserve_evaluation=preserve)
    assert get_callable_args(region.expression, getattr(Scalar, operation)) is not None
    assert region.originals == (z,)
    assert decode(region.expression, region) is original
    result = optimise(original, OptimisationOptions(preserve_floating_point_evaluation=preserve))
    assert result.report.regions
    assert_values(original, result.expression, {})


@pytest.mark.parametrize("preserve", [False, True])
def test_complex_rules_run_in_egglog_without_changing_arithmetic_in_preserving_mode(preserve):
    from uflx.arithmetic import conjugate, real_part

    x = RuntimeArray("real-unary", ())
    arithmetic = (x + num(1.0)) + num(-1.0)
    original = real_part(conjugate(conjugate(arithmetic)))
    result = optimise(original, OptimisationOptions(preserve_floating_point_evaluation=preserve))
    assert any(r.changed for r in result.report.regions)
    if preserve:
        assert result.expression is arithmetic
    assert_values(original, result.expression, {"real-unary": np.array(0.25)})


@pytest.mark.parametrize("preserve", [False, True])
def test_complex_rules_do_not_treat_complex_values_as_real(preserve):
    from uflx.arithmetic import imaginary_part, real_part

    z = num(2 + 3j)
    for original in (real_part(z), imaginary_part(z)):
        result = optimise(
            original, OptimisationOptions(preserve_floating_point_evaluation=preserve)
        )
        assert result.expression is original
        assert result.expression.scalar_type == "real"
        assert_values(original, result.expression, {})


@pytest.mark.parametrize("preserve", [False, True])
def test_complex_zero_rewrites_preserve_real_type_and_free_indices(preserve):
    from uflx.arithmetic import imaginary_part, real_part

    i = index("complex-zero-index")
    zero = Zero((), ((i, 3),), (), "complex")
    for original in (real_part(zero) + num(0), imaginary_part(real_part(zero))):
        result = optimise(
            original, OptimisationOptions(preserve_floating_point_evaluation=preserve)
        )
        assert_values(original, result.expression, {})


@pytest.mark.parametrize("preserve", [False, True])
def test_elementary_scalar_rules_and_evaluation_policy(preserve):
    from uflx.arithmetic import absolute, logarithm

    options = OptimisationOptions(preserve_floating_point_evaluation=preserve)
    x = RuntimeArray("power-one", ())
    for original, expected in (
        (x ** num(1), x),
        (num(0) ** num(0), num(1)),
        (logarithm(num(1)), num(0)),
    ):
        result = optimise(original, options).expression
        if preserve:
            assert result is original
        else:
            assert result is expected or (expected is num(0) and isinstance(result, Zero))
        assert_values(original, result, {"power-one": np.array(-3.5)})
    z = num(3 + 4j)
    original = absolute(absolute(z))
    result = optimise(original, options)
    assert result.expression is absolute(z)
    assert any(r.changed for r in result.report.regions)
    assert_values(original, result.expression, {})


def test_power_one_preserves_complex_type_and_family_indices():
    i = index("power-family")
    a = tensor((num(1j), num(2j)))
    original = a[i] ** num(1)
    result = optimise(original).expression
    assert result.scalar_type == "complex"
    assert_values(original, result, {})


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("fixed", [False, True])
def test_component_tensor_selection_of_runtime_vector(preserve, fixed):
    i, j = index("component-i"), index("component-j")
    a = RuntimeArray("component-vector", (3,))
    entry = 1 if fixed else j
    original = component_tensor(a[i], (i,))[entry]
    result = optimise(original, OptimisationOptions(preserve_floating_point_evaluation=preserve))
    assert result.expression is a[entry]
    assert_values(original, result.expression, {"component-vector": np.arange(3.0)})


@pytest.mark.parametrize("preserve", [False, True])
def test_component_tensor_selection_substitutes_indices_simultaneously(preserve):
    i, j = index("swap-i"), index("swap-j")
    a = RuntimeArray("swap-matrix", (3, 3))
    selected = component_tensor(a[i, j], (j, i))[i, j]
    original = component_tensor(selected, (i, j))
    result = optimise(original, OptimisationOptions(preserve_floating_point_evaluation=preserve))
    assert result.expression.expression is a[j, i]
    assert_values(original, result.expression, {"swap-matrix": np.arange(9.0).reshape(3, 3)})


@pytest.mark.parametrize("preserve", [False, True])
def test_component_tensor_selection_avoids_capture(preserve):
    i, j = index("capture-i"), index("capture-j")
    a = tensor((tensor((num(1), num(2))), tensor((num(3), num(5)))))
    body = index_sum(a[i, j], (j,))
    original = component_tensor(body, (i,))[j]
    result = optimise(original, OptimisationOptions(preserve_floating_point_evaluation=preserve))
    assert isinstance(result.expression, IndexSum)
    assert result.expression.indices != (j,)
    assert_values(original, result.expression, {})


@pytest.mark.parametrize("preserve", [False, True])
def test_exact_rational_collection_and_evaluation_policy(preserve):
    x = RuntimeArray("rational-x", ())
    original = x / num(2) + x / num(3)
    result = optimise(
        original, OptimisationOptions(iterations=5, preserve_floating_point_evaluation=preserve)
    ).expression
    if preserve:
        assert result is original
    else:
        assert result in ((num(5) / num(6)) * x, x * (num(5) / num(6)))
    assert_values(original, result, {"rational-x": np.array(1.25)})


@pytest.mark.parametrize("denominator", [0, 2, -3])
def test_rational_constant_folding_retains_zero_denominators(denominator):
    original = num(6) / num(denominator)
    result = optimise(original).expression
    if denominator == 0:
        assert result is original
        assert num(1) / num(0) in nodes(optimise(num(0) * (num(1) / num(0))).expression)
    else:
        assert result is num(3 if denominator == 2 else -2)
        assert_values(original, result, {})


def test_rational_cancellation_preserves_unknown_singularities():
    x = RuntimeArray("rational-singularity", ())
    for original in (x / x, num(0) / x, num(0) * (num(1) / x)):
        result = optimise(original).expression
        assert any(n is x for n in nodes(result))
        assert not isinstance(result, Zero)
        assert_values(original, result, {"rational-singularity": np.array(2.0)})


def test_rational_collection_preserves_complex_values_and_indices():
    i = index("rational-family")
    z = tensor((num(1j), num(2 + 3j)))[i]
    original = component_tensor(z / num(2) + z / num(3), (i,))
    result = optimise(original, OptimisationOptions(iterations=5)).expression
    assert_values(original, result, {})


def test_nested_rational_denominators_and_derived_zero():
    from uflx._src.nodes import Division

    x = RuntimeArray("nested-rational", ())
    original = x / (num(1) / num(3))
    result = optimise(original).expression
    assert result in (num(3) * x, x * num(3))
    assert_values(original, result, {"nested-rational": np.array(2.0)})
    undefined = num(1) / (num(1) - num(1))
    result = optimise(undefined).expression
    assert isinstance(result, Division)
    assert isinstance(result.denominator, Zero) or result.denominator is num(0)


def test_zero_product_does_not_assume_arbitrary_matrix_inverses_exist():
    from uflx._src.nodes import Inverse
    from uflx.tensors import inverse

    A = RuntimeArray("possibly-singular", (2, 2))
    original = num(0) * inverse(A)[0, 0]
    result = optimise(original).expression
    assert any(isinstance(n, Inverse) for n in nodes(result))
    assert_values(original, result, {"possibly-singular": np.eye(2)})


@pytest.mark.parametrize("size", [0, 1, 2, 3])
@pytest.mark.parametrize("preserve", [False, True])
def test_constant_identity_matrices(size, preserve):
    from uflx._src.nodes import Determinant, Inverse, Logarithm
    from uflx.arithmetic import logarithm
    from uflx.tensors import determinant, inverse

    matrix = identity(size)
    options = OptimisationOptions(preserve_floating_point_evaluation=preserve)
    for original in (determinant(matrix), inverse(matrix), logarithm(determinant(matrix))):
        result = optimise(original, options).expression
        if preserve:
            assert result is original
        else:
            assert not any(isinstance(n, Determinant | Inverse | Logarithm) for n in nodes(result))
        assert_values(original, result, {})


@pytest.mark.parametrize("values", [((2,),), ((2, 1), (1, 3)), ((0, 2, 1), (1, 0, 3), (4, 1, 2))])
def test_exact_small_matrix_determinants_and_inverses(values):
    from uflx._src.nodes import Determinant, Float, Inverse
    from uflx.tensors import determinant, inverse

    matrix = tensor(tuple(tensor(tuple(num(v) for v in row)) for row in values))
    for original in (determinant(matrix), inverse(matrix)):
        result = optimise(original).expression
        assert not any(isinstance(n, Determinant | Inverse | Float) for n in nodes(result))
        assert_values(original, result, {})


def test_matrix_inverse_keeps_small_exact_rational_pivots():
    from uflx.tensors import inverse

    tiny = num(1) / num(10**30)
    matrix = tensor((tensor((tiny, num(0))), tensor((num(0), num(1)))))
    result = optimise(inverse(matrix)).expression
    assert result.entries[0].entries[0] is num(10**30)
    np.testing.assert_allclose(
        evaluate(inverse(matrix), {}), np.asarray(evaluate(result, {}), dtype=float), rtol=1e-12
    )


@pytest.mark.parametrize("values", [((0,),), ((1, 2), (2, 4)), ((0, 0), (0, 0))])
def test_singular_constant_matrix_inverse_stays_undefined(values):
    from uflx._src.nodes import Inverse
    from uflx.tensors import determinant, inverse

    matrix = tensor(tuple(tensor(tuple(num(v) for v in row)) for row in values))
    det = optimise(determinant(matrix)).expression
    assert isinstance(det, Zero) or det is num(0)
    assert_values(determinant(matrix), det, {})
    for original in (inverse(matrix), num(0) * inverse(matrix)[0, 0]):
        result = optimise(original).expression
        assert any(isinstance(n, Inverse) for n in nodes(result))


def test_constant_matrix_fold_preserves_family_and_complex_zero_type():
    from uflx.tensors import determinant

    i = index("matrix-family")
    for size in (0, 2):
        matrix = Zero((size, size), ((i, 3),), (), "complex")
        original = component_tensor(determinant(matrix), (i,))
        result = optimise(original).expression
        assert_values(original, result, {})


@pytest.mark.parametrize("operation", ["determinant", "inverse", "logdet"])
@pytest.mark.parametrize("preserve", [False, True])
def test_constant_matrix_families_through_broadcast(operation, preserve):
    from uflx._src.nodes import Determinant, Inverse, Logarithm
    from uflx.arithmetic import logarithm
    from uflx.tensors import broadcast_to, determinant, inverse

    i = index("repeated-matrix")
    matrix = broadcast_to(identity(2), (3, 2, 2))[i]
    expression = {
        "determinant": determinant(matrix),
        "inverse": inverse(matrix)[0, 0],
        "logdet": logarithm(determinant(matrix)),
    }[operation]
    original = component_tensor(expression, (i,))
    result = optimise(
        original, OptimisationOptions(preserve_floating_point_evaluation=preserve)
    ).expression
    assert any(isinstance(n, Determinant | Inverse | Logarithm) for n in nodes(result)) == preserve
    assert_values(original, result, {})


@pytest.mark.parametrize("reverse", [False, True])
@pytest.mark.parametrize("compound", [False, True])
def test_delta_contracts_matrix_entries_and_compound_summands(reverse, compound):
    i, j, k = (index(name) for name in ("delta-i", "delta-j", "delta-k"))
    matrix = tensor(tuple(tensor(tuple(num(1 + 3 * r + c) for c in range(3))) for r in range(3)))
    body = matrix[j, k] + num(2) * matrix[k, j] if compound else matrix[j, k]
    delta = identity(3)[j, i] if reverse else identity(3)[i, j]
    original = component_tensor(index_sum(delta * body, (j,)), (i, k))
    result = optimise(original).expression
    assert not any(isinstance(n, IndexSum) for n in nodes(result))
    assert_values(original, result, {})


def test_delta_contraction_avoids_capture_by_nested_sum():
    i, j = index("capture-target"), index("capture-removed")
    matrix = tensor((tensor((num(1), num(2))), tensor((num(4), num(8)))))
    inner = index_sum(matrix[j, i], (i,))
    original = component_tensor(index_sum(identity(2)[i, j] * inner, (j,)), (i,))
    result = optimise(original).expression
    np.testing.assert_allclose(evaluate(result, {}), [3, 12])
    assert sum(isinstance(n, IndexSum) for n in nodes(result)) == 1
    assert_values(original, result, {})


def test_delta_contracts_one_axis_of_multi_index_sum():
    i, j, k = (index(name) for name in ("multi-i", "multi-j", "multi-k"))
    matrix = tensor((tensor((num(1), num(2))), tensor((num(4), num(8)))))
    original = component_tensor(index_sum(identity(2)[i, j] * matrix[j, k], (j, k)), (i,))
    result = optimise(original).expression
    sums = [n for n in nodes(result) if isinstance(n, IndexSum)]
    assert len(sums) == 1 and len(sums[0].indices) == 1
    assert_values(original, result, {})


def test_delta_contraction_retains_other_bound_axis_when_both_are_summed():
    i, j = index("both-i"), index("both-j")
    original = index_sum(identity(3)[i, j], (i, j))
    result = optimise(original).expression
    assert_values(original, result, {})
    assert float(evaluate(result, {})) == 3
    assert not any(isinstance(n, IndexSum) and len(n.indices) == 2 for n in nodes(result))


@pytest.mark.parametrize("preserve, groups", [(True, ("indexed",)), (False, ("arithmetic",))])
def test_general_delta_contraction_respects_policy(preserve, groups):
    i, j, k = (index(name) for name in ("policy-i", "policy-j", "policy-k"))
    matrix = tensor((tensor((num(1), num(2))), tensor((num(4), num(8)))))
    original = component_tensor(index_sum(identity(2)[i, j] * matrix[j, k], (j,)), (i, k))
    result = optimise(
        original,
        OptimisationOptions(rule_groups=groups, preserve_floating_point_evaluation=preserve),
    ).expression
    assert any(isinstance(n, IndexSum) for n in nodes(result))
    assert_values(original, result, {})


def test_general_delta_contraction_distinguishes_runtime_and_undefined_entries():
    i, j, k = (index(name) for name in ("unknown-i", "unknown-j", "unknown-k"))
    for matrix in (
        RuntimeArray("matrix", (2, 2)),
        tensor((tensor((num(1), num(1) / num(0))), tensor((num(2), num(3))))),
    ):
        original = component_tensor(index_sum(identity(2)[i, j] * matrix[j, k], (j,)), (i, k))
        result = optimise(original).expression
        assert any(isinstance(n, IndexSum) for n in nodes(result)) == (not matrix.is_terminal)
        if matrix.is_terminal:
            assert_values(original, result, {"matrix": np.arange(4.0).reshape(2, 2)})


def test_oversized_region_optimises_children_and_retries_smaller_parent():
    x, y = RuntimeArray("budget-x", ()), RuntimeArray("budget-y", ())
    original = (x + num(0)) * (y + num(0))
    result = optimise(original, OptimisationOptions(max_region_nodes=3))
    assert result.report.regions[0].search_stop == "region_limit"
    assert any(r.changed for r in result.report.regions)
    assert result.expression in (x * y, y * x)
    assert all(r.nodes <= 3 for r in result.report.regions if r.search_stop != "region_limit")
    assert_values(
        original, result.expression, {"budget-x": np.array(2.0), "budget-y": np.array(3.0)}
    )


def test_oversized_region_reuses_a_shared_child_search(monkeypatch):
    from uflx._src.optimisation import engine

    searched = []
    run_region = engine.run_region

    def record_search(expression, *args):
        searched.append(expression)
        return run_region(expression, *args)

    monkeypatch.setattr(engine, "run_region", record_search)
    x, y = RuntimeArray("shared-budget-x", ()), RuntimeArray("shared-budget-y", ())
    shared = x / num(2) + x / num(3)
    original = shared * y + shared * (y + num(1))
    result = optimise(original, OptimisationOptions(max_region_nodes=5, iterations=5))
    assert searched.count(shared) == 1
    assert any(r.search_stop == "region_limit" for r in result.report.regions)
    assert any(r.changed for r in result.report.regions)
    assert_values(
        original,
        result.expression,
        {"shared-budget-x": np.array(2.0), "shared-budget-y": np.array(3.0)},
    )


@pytest.mark.parametrize("groups", [("arithmetic",), ("herbie",)])
@pytest.mark.parametrize("preserve", [False, True])
def test_runtime_polynomial_cancellation_respects_fp_policy(groups, preserve):
    x = RuntimeArray("defined-polynomial", ())
    polynomial = x * x + num(2) * x
    expression = polynomial - polynomial
    result = optimise(
        expression,
        OptimisationOptions(rule_groups=groups, preserve_floating_point_evaluation=preserve),
    ).expression
    if preserve:
        assert result is expression
    else:
        assert isinstance(result, Zero)
    assert_values(expression, result, {"defined-polynomial": np.array(3.0)})
