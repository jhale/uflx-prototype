"""Compare indexed extraction against the compiler's independent array evaluator."""

import numpy as np
import pytest

pytest.importorskip("egglog")

from uflx_compiler.evaluation import evaluate
from uflx_spec_interpreter import component_tensor, index, index_sum, num, tensor
from uflx_spec_interpreter._src.nodes import IndexSum, Zero
from uflx_spec_interpreter._src.nodes.structure import expression_operands
from uflx_spec_interpreter._src.optimisation import OptimisationOptions, optimise
from uflx_spec_interpreter._src.tensors import identity
from uflx_toy.nodes import RuntimeArray


def optimise_core(expression, iterations=3, max_region=60, loop_aware=True):
    result = optimise(
        expression,
        OptimisationOptions(
            iterations=iterations,
            max_region_nodes=max_region,
            cost="loops" if loop_aware else "tree",
        ),
    )
    return result.expression, result.report


def nodes(root):
    seen, pending = set(), [root]
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(expression_operands(node))
    return seen


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


def test_region_budget_skips_without_changing_values():
    i = index("i")
    a = RuntimeArray("a", (4,))
    original = index_sum(a[i] + a[i] * num(2), (i,))
    result, stats = optimise_core(original, max_region=1)
    assert sum(region.search_stop == "region_limit" for region in stats.regions) == 1
    assert result is original


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
    from uflx_spec_interpreter._src.nodes import Product

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
    from uflx_spec_interpreter._src.algorithms.binding import alpha_equivalent

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


def test_alpha_atom_buckets_check_collisions_and_preserve_free_identities(monkeypatch):
    from uflx_spec_interpreter import function
    from uflx_spec_interpreter._src.algorithms.alpha import AlphaHasher
    from uflx_spec_interpreter._src.optimisation.encoding import decode, encode
    from uflx_toy.spaces import space

    V = space("alpha-atom", scalar_type="real")
    u, v = (function(V, name) for name in ("u", "v"))
    i, j = index("i"), index("j")
    a = RuntimeArray("alpha-array", (3,))
    left = index_sum(u * a[i], (i,))
    renamed = index_sum(u * a[j], (j,))
    different = index_sum(v * a[i], (i,))
    monkeypatch.setattr(AlphaHasher, "fingerprint", lambda self, expression: 0)
    region = encode(left + renamed + different + u + v, encode_indices=False)
    assert region.originals == (left, different, u, v)
    assert region.alpha_merges == 1
    assert decode(region.expression, region) is left + left + different + u + v


def test_codec_roundtrip_preserves_original_opaque_objects():
    from uflx_spec_interpreter._src.optimisation.encoding import decode, encode

    node = RuntimeArray("opaque-reference", ())
    region = encode(node, encode_indices=False)
    # A second encoding must not change the first region's reconstruction table.
    other = encode(RuntimeArray("other-reference", ()), encode_indices=False)
    assert region.originals != other.originals
    assert decode(region.expression, region) is node


def test_options_and_search_reports():
    from uflx_spec_interpreter._src.optimisation import OptimisationOptions

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
    from uflx_spec_interpreter._src.optimisation.costs import loop_cost

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
    from uflx_compiler.algebra import _normalizer
    from uflx_oracle import evaluate as oracle
    from uflx_spec_interpreter import apply_derivatives, directional_derivative, function, lower
    from uflx_toy.spaces import space

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
    from uflx_spec_interpreter._src.nodes import Product, Sum

    arithmetic = lambda e: sum(isinstance(n, Sum | Product) for n in nodes(e))
    assert arithmetic(result) < arithmetic(cleaned)
    for seed in range(10):
        rng = np.random.default_rng(seed)
        env = {x: rng.uniform(-0.8, 0.8, size=x.value_shape) for x in (u, v, w, h)}
        expected = oracle(derivative, env)(())
        np.testing.assert_allclose(oracle(result, env)(()), expected, rtol=1e-10, atol=1e-10)


def test_complex_grouped_form_and_rebind():
    from uflx_oracle import evaluate as oracle
    from uflx_spec_interpreter import conjugate, form, function
    from uflx_spec_interpreter._src.algorithms.binding import alpha_equivalent
    from uflx_spec_interpreter._src.extensions.binding import rebind
    from uflx_toy.spaces import space

    V = space("optimisation-complex", scalar_type="complex")
    u, v, w, p = (function(V, name) for name in ("u", "v", "w", "p"))
    original = form(u * conjugate(v) + u * conjugate(w), ((u, conjugate(v)),))
    result = optimise(original).expression
    assert result.arguments == original.arguments
    np.testing.assert_allclose(
        oracle(original, {w: 3j})(((2j, 1 + 4j),)),
        oracle(result, {w: 3j})(((2j, 1 + 4j),)),
    )
    a = form(num(2) * u + num(3) * u, (u,))
    assert alpha_equivalent(optimise(a).expression, optimise(rebind(a, {u: p})).expression)


def test_coefficient_cancellation_retains_unknown_operand():
    a = RuntimeArray("cancel", ())
    original = num(2) * a + num(-2) * a
    result = optimise(original, OptimisationOptions(iterations=4)).expression
    assert not isinstance(result, Zero)
    assert a in nodes(result)
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


@pytest.mark.parametrize("denominator", [0, 2])
def test_zero_product_uses_literal_definedness(denominator):
    partial = num(1) / num(denominator)
    result = optimise(num(0) * partial).expression
    assert isinstance(result, Zero) == (denominator != 0)
    if denominator == 0:
        assert partial in nodes(result)


def test_delta_does_not_discard_unknown_vector_components():
    i, j = index("i"), index("j")
    a = RuntimeArray("unknown-vector", (3,))
    original = component_tensor(index_sum(identity(3)[i, j] * a[j], (j,)), (i,))
    result = optimise(original).expression
    assert any(isinstance(n, IndexSum) for n in nodes(result))
    assert_values(original, result, {"unknown-vector": np.arange(3.0)})


@pytest.mark.parametrize("preserve_floating_point_evaluation", [False, True])
def test_empty_sum_does_not_expose_undefined_factor(preserve_floating_point_evaluation):
    from uflx_spec_interpreter import broadcast_to

    i = index("empty")
    partial = num(1) / num(0)
    body = broadcast_to(num(2), (0,))[i] * partial
    original = index_sum(body, (i,))
    result = optimise(
        original,
        OptimisationOptions(preserve_floating_point_evaluation=preserve_floating_point_evaluation),
    ).expression
    assert isinstance(result, Zero)


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
    from uflx_spec_interpreter import broadcast_to

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


@pytest.mark.parametrize("preserve_floating_point_evaluation", [False, True])
def test_complex_simplifications_preserve_undefined_operands(preserve_floating_point_evaluation):
    from uflx_spec_interpreter import conjugate, imaginary_part, real_part

    options = OptimisationOptions(
        preserve_floating_point_evaluation=preserve_floating_point_evaluation
    )
    x = RuntimeArray("complex-real", ())
    partial = num(1) / x
    assert optimise(conjugate(conjugate(partial)), options).expression is partial
    assert optimise(real_part(partial), options).expression is partial
    unknown_imaginary = imaginary_part(partial)
    assert optimise(unknown_imaginary, options).expression is unknown_imaginary
    assert isinstance(optimise(imaginary_part(num(1) / num(2)), options).expression, Zero)


def test_reject_nonboolean_fp_policy():
    with pytest.raises(ValueError, match="preserve_floating_point_evaluation"):
        OptimisationOptions(preserve_floating_point_evaluation="no")


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


def test_preserving_mode_does_not_insert_complex_promotion_arithmetic():
    original = tensor((num(-0.0), num(1j)))[0]
    result = optimise(
        original, OptimisationOptions(preserve_floating_point_evaluation=True)
    ).expression
    assert result is original


def test_unknown_zero_product_retains_family_indices_without_rewrite_cycle():
    i, j = index("i"), index("j")
    a = RuntimeArray("family-array", (3,))
    original = component_tensor(
        index_sum(Zero((), ((i, 2), (j, 3)), (), "real") * a[j], (j,)), (i,)
    )
    result = optimise(original).expression
    assert a in nodes(result)
    assert_values(original, result, {"family-array": np.arange(3.0)})


@pytest.mark.parametrize("preserve", [False, True])
@pytest.mark.parametrize("operation", ["conjugate", "real_part", "imaginary_part"])
def test_complex_operators_roundtrip_through_egglog(operation, preserve):
    from egglog.deconstruct import get_callable_args

    from uflx_spec_interpreter import conjugate, imaginary_part, real_part
    from uflx_spec_interpreter._src.optimisation.encoding import decode, encode
    from uflx_spec_interpreter._src.optimisation.language import Scalar

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
    from uflx_spec_interpreter import conjugate, real_part

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
    from uflx_spec_interpreter import imaginary_part, real_part

    z = num(2 + 3j)
    for original in (real_part(z), imaginary_part(z)):
        result = optimise(
            original, OptimisationOptions(preserve_floating_point_evaluation=preserve)
        )
        assert result.expression is original
        assert result.expression.scalar_type == "real"
        assert_values(original, result.expression, {})


@pytest.mark.parametrize("preserve", [False, True])
def test_double_conjugation_of_tensor_and_form_atoms(preserve):
    from uflx_spec_interpreter import conjugate, form, function
    from uflx_toy.spaces import space

    u = function(space("complex-form-atom", scalar_type="complex"), "u")
    options = OptimisationOptions(preserve_floating_point_evaluation=preserve)
    for value in (tensor((num(1j), num(2j))), form(u, (u,))):
        result = optimise(conjugate(conjugate(value)), options)
        assert result.expression is value
        assert any(r.changed for r in result.report.regions)


@pytest.mark.parametrize("preserve", [False, True])
def test_complex_zero_rewrites_preserve_real_type_and_free_indices(preserve):
    from uflx_spec_interpreter import imaginary_part, real_part

    i = index("complex-zero-index")
    zero = Zero((), ((i, 3),), (), "complex")
    for original in (real_part(zero) + num(0), imaginary_part(real_part(zero))):
        result = optimise(
            original, OptimisationOptions(preserve_floating_point_evaluation=preserve)
        )
        assert_values(original, result.expression, {})


def test_complex_search_obeys_region_budget():
    from uflx_spec_interpreter import conjugate

    x = RuntimeArray("complex-budget", ())
    original = conjugate(conjugate(x))
    result = optimise(original, OptimisationOptions(max_region_nodes=1))
    assert result.expression is original
    assert result.report.regions[0].search_stop == "region_limit"


@pytest.mark.parametrize("preserve", [False, True])
def test_elementary_scalar_rules_and_evaluation_policy(preserve):
    from uflx_spec_interpreter import absolute, logarithm

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


@pytest.mark.parametrize("denominator", [0, 2])
def test_power_zero_requires_a_defined_base(denominator):
    partial = num(1) / num(denominator)
    original = partial ** num(0)
    result = optimise(original).expression
    if denominator:
        assert result is num(1)
    else:
        assert partial in nodes(result)
        assert result is original
    unknown = RuntimeArray("unknown-power-base", ()) ** num(0)
    assert optimise(unknown).expression is unknown


def test_nested_absolute_keeps_undefined_operand():
    from uflx_spec_interpreter import absolute

    partial = num(1) / num(0)
    assert optimise(absolute(absolute(partial))).expression is absolute(partial)


def test_logarithm_only_folds_one():
    from uflx_spec_interpreter import logarithm

    for value in (0, -1, 2):
        original = logarithm(num(value))
        assert optimise(original).expression is original


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
def test_component_selection_does_not_discard_unknown_singularities(preserve):
    i, j = index("partial-i"), index("partial-j")
    a = RuntimeArray("partial-vector", (3,))
    original = component_tensor(num(1) / a[i], (i,))[j]
    result = optimise(original, OptimisationOptions(preserve_floating_point_evaluation=preserve))
    assert result.expression is original


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


def test_rational_factors_cancel_without_assuming_the_operand_is_defined():
    x = RuntimeArray("rational-cancel", ())
    for operand in (x, num(1) / x):
        original = (num(2) * operand) / num(2)
        assert optimise(original).expression is operand


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


def test_rational_arithmetic_preserves_arbitrary_precision():
    from uflx_spec_interpreter._src.nodes import Division, Integer

    large = 2**100
    original = num(large) / num(3) + num(1) / num(3)
    result = optimise(original).expression
    assert isinstance(result, Division)
    assert result.first is num(large + 1)
    assert result.second is num(3)
    assert all(isinstance(n, Integer | Division) for n in nodes(result))


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
    from uflx_spec_interpreter._src.nodes import Division

    x = RuntimeArray("nested-rational", ())
    original = x / (num(1) / num(3))
    result = optimise(original).expression
    assert result in (num(3) * x, x * num(3))
    assert_values(original, result, {"nested-rational": np.array(2.0)})
    undefined = num(1) / (num(1) - num(1))
    result = optimise(undefined).expression
    assert isinstance(result, Division)
    assert isinstance(result.second, Zero) or result.second is num(0)


def test_floating_point_divisors_remain_opaque():
    x = RuntimeArray("float-divisor", ())
    original = x / num(2.0)
    assert optimise(original).expression is original


def test_zero_product_does_not_assume_arbitrary_matrix_inverses_exist():
    from uflx_spec_interpreter import inverse
    from uflx_spec_interpreter._src.nodes import Inverse

    A = RuntimeArray("possibly-singular", (2, 2))
    original = num(0) * inverse(A)[0, 0]
    result = optimise(original).expression
    assert any(isinstance(n, Inverse) for n in nodes(result))
    assert_values(original, result, {"possibly-singular": np.eye(2)})
