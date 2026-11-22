"""Optimisation of runtime arrays and symbolic extents."""

import pytest

pytest.importorskip("egglog")

from uflx._src.nodes import Zero
from uflx._src.optimisation import OptimisationOptions, optimise
from uflx._src.tensors import identity
from uflx.arithmetic import num
from uflx.tensors import component_tensor, index, index_sum, tensor
from uflx_basix.nodes import RuntimeArray

from ..uflx.test_optimisation import nodes


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


def test_region_budget_retains_expression_when_no_subregion_fits():
    i = index("i")
    a = RuntimeArray("a", (4,))
    original = index_sum(a[i] + a[i] * num(2), (i,))
    result, stats = optimise_core(original, max_region=1)
    assert stats.regions and all(region.search_stop == "region_limit" for region in stats.regions)
    assert result is original


def test_alpha_atom_buckets_check_collisions_and_preserve_free_identities(monkeypatch):
    from uflx._src.algorithms.alpha import AlphaHasher
    from uflx._src.optimisation.encoding import decode, encode
    from uflx.forms import function
    from uflx_basix.spaces import space

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
    from uflx._src.optimisation.encoding import decode, encode

    node = RuntimeArray("opaque-reference", ())
    region = encode(node, encode_indices=False)
    # A second encoding must not change the first region's reconstruction table.
    other = encode(RuntimeArray("other-reference", ()), encode_indices=False)
    assert region.originals != other.originals
    assert decode(region.expression, region) is node


@pytest.mark.parametrize("preserve_floating_point_evaluation", [False, True])
def test_complex_simplifications_preserve_undefined_operands(preserve_floating_point_evaluation):
    from uflx.arithmetic import conjugate, imaginary_part, real_part

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


def test_complex_search_obeys_region_budget():
    from uflx.arithmetic import conjugate

    x = RuntimeArray("complex-budget", ())
    original = conjugate(conjugate(x))
    result = optimise(original, OptimisationOptions(max_region_nodes=1))
    assert result.expression is original
    assert result.report.regions[0].search_stop == "region_limit"


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
    assert optimise(unknown).expression is num(1)


@pytest.mark.parametrize("preserve", [False, True])
def test_component_selection_does_not_discard_unknown_singularities(preserve):
    i, j = index("partial-i"), index("partial-j")
    a = RuntimeArray("partial-vector", (3,))
    original = component_tensor(num(1) / a[i], (i,))[j]
    result = optimise(original, OptimisationOptions(preserve_floating_point_evaluation=preserve))
    assert result.expression is original


def test_rational_factors_cancel_without_assuming_the_operand_is_defined():
    x = RuntimeArray("rational-cancel", ())
    for operand in (x, num(1) / x):
        original = (num(2) * operand) / num(2)
        assert optimise(original).expression is operand


def test_floating_point_divisors_remain_opaque():
    x = RuntimeArray("float-divisor", ())
    original = x / num(2.0)
    assert optimise(original).expression is original


def test_matrix_folding_leaves_unknown_float_rectangular_and_large_inputs():
    from uflx._src.nodes import Determinant, Inverse
    from uflx.tensors import determinant, inverse

    matrices = (
        RuntimeArray("matrix", (2, 2)),
        identity(4),
        tensor((tensor((num(1.0), num(0))), tensor((num(0), num(2))))),
    )
    for matrix in matrices:
        for operation, node_type in ((determinant, Determinant), (inverse, Inverse)):
            result = optimise(operation(matrix)).expression
            assert isinstance(result, node_type)
    rectangular = tensor((tensor((num(1),)), tensor((num(2),))))
    assert isinstance(optimise(inverse(rectangular)).expression, Inverse)


def test_defined_runtime_input_does_not_make_logarithm_total():
    from uflx.arithmetic import logarithm

    x = RuntimeArray("log-input", ())
    partial = logarithm(x)
    for expression in (num(0) * partial, partial - partial):
        result = optimise(expression, OptimisationOptions(rule_groups=("arithmetic",))).expression
        assert partial in nodes(result)
