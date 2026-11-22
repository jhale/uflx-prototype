"""Symbolic identities, definedness and binding in the optimiser."""

import pytest

pytest.importorskip("egglog")

from uflx._src.nodes import Zero
from uflx._src.nodes.structure import expression_operands
from uflx._src.optimisation import OptimisationOptions, optimise
from uflx._src.tensors import identity
from uflx.arithmetic import num
from uflx.tensors import index, index_sum, tensor


def nodes(root):
    seen, pending = set(), [root]
    while pending:
        node = pending.pop()
        if node not in seen:
            seen.add(node)
            pending.extend(expression_operands(node))
    return seen


@pytest.mark.parametrize("denominator", [0, 2])
def test_zero_product_uses_literal_definedness(denominator):
    partial = num(1) / num(denominator)
    result = optimise(num(0) * partial).expression
    assert isinstance(result, Zero) == (denominator != 0)
    if denominator == 0:
        assert partial in nodes(result)


@pytest.mark.parametrize("preserve_floating_point_evaluation", [False, True])
def test_empty_sum_does_not_expose_undefined_factor(preserve_floating_point_evaluation):
    from uflx.tensors import broadcast_to

    i = index("empty")
    partial = num(1) / num(0)
    body = broadcast_to(num(2), (0,))[i] * partial
    original = index_sum(body, (i,))
    result = optimise(
        original,
        OptimisationOptions(preserve_floating_point_evaluation=preserve_floating_point_evaluation),
    ).expression
    assert isinstance(result, Zero)


def test_reject_nonboolean_fp_policy():
    with pytest.raises(ValueError, match="preserve_floating_point_evaluation"):
        OptimisationOptions(preserve_floating_point_evaluation="no")


def test_preserving_mode_does_not_insert_complex_promotion_arithmetic():
    original = tensor((num(-0.0), num(1j)))[0]
    result = optimise(
        original, OptimisationOptions(preserve_floating_point_evaluation=True)
    ).expression
    assert result is original


@pytest.mark.parametrize("preserve", [False, True])
def test_double_conjugation_of_tensor_and_form_atoms(preserve):
    from uflx.arithmetic import conjugate
    from uflx.forms import form, function

    from .spaces import space

    u = function(space("complex-form-atom", scalar_type="complex"), "u")
    options = OptimisationOptions(preserve_floating_point_evaluation=preserve)
    for value in (tensor((num(1j), num(2j))), form(u, (u,))):
        result = optimise(conjugate(conjugate(value)), options)
        assert result.expression is value
        assert any(r.changed for r in result.report.regions)


def test_nested_absolute_keeps_undefined_operand():
    from uflx.arithmetic import absolute

    partial = num(1) / num(0)
    assert optimise(absolute(absolute(partial))).expression is absolute(partial)


def test_logarithm_only_folds_one():
    from uflx.arithmetic import logarithm

    for value in (0, -1, 2):
        original = logarithm(num(value))
        assert optimise(original).expression is original


def test_rational_arithmetic_preserves_arbitrary_precision():
    from uflx._src.nodes import Division, Integer

    large = 2**100
    original = num(large) / num(3) + num(1) / num(3)
    result = optimise(original).expression
    assert isinstance(result, Division)
    assert result.numerator is num(large + 1)
    assert result.denominator is num(3)
    assert all(isinstance(n, Integer | Division) for n in nodes(result))


def test_matrix_fold_does_not_discard_undefined_entries_even_for_empty_matrices():
    from uflx._src.nodes import Determinant, Inverse
    from uflx.tensors import broadcast_to, determinant, inverse

    for size in (0, 2):
        matrix = broadcast_to(num(1) / num(0), (size, size))
        for operation, node_type in ((determinant, Determinant), (inverse, Inverse)):
            assert isinstance(optimise(operation(matrix)).expression, node_type)


def test_constant_inverse_produces_exact_non_dyadic_fractions():
    from uflx.tensors import inverse

    matrix = tensor((tensor((num(2), num(1))), tensor((num(1), num(3)))))
    expected = tensor(
        (tensor((num(3) / num(5), num(-1) / num(5))), tensor((num(-1) / num(5), num(2) / num(5))))
    )
    assert optimise(inverse(matrix)).expression is expected


def test_constant_matrix_selection_does_not_discard_undefined_siblings():
    from uflx.tensors import determinant

    undefined = num(1) / num(0)
    invalid = tensor((tensor((undefined, num(0))), tensor((num(0), num(1)))))
    original = determinant(tensor((identity(2), invalid))[0])
    assert undefined in nodes(optimise(original).expression)


def test_definedness_propagates_to_new_egglog_arithmetic():
    from egglog import BigRat, EGraph

    from uflx._src.optimisation.language import Scalar
    from uflx._src.optimisation.rules import constant_cleanup, defined

    x, y, z = (Scalar.atom(i) for i in range(3))
    polynomial = (x + y) * z
    quotient = polynomial / Scalar.number(BigRat(2, 1))
    graph = EGraph()
    # Only terminals have facts: these compound terms need the e-graph analysis.
    graph.register(quotient, defined(x), defined(y), defined(z))
    graph.run(constant_cleanup * 4)
    graph.check(defined(polynomial), defined(quotient))
