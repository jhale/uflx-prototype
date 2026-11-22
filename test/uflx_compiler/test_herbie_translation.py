"""Check source provenance, conservative translation and executable Herbie identities."""

import pytest


def test_herbie_group_is_opt_in_and_respects_floating_point_policy():
    pytest.importorskip("egglog")
    from uflx._src.optimisation import OptimisationOptions, optimise
    from uflx.arithmetic import num
    from uflx_basix.runtime import runtime_array
    from uflx_compiler.evaluation import evaluate

    x = runtime_array("herbie-x", ())
    expression = num(3) * x + num(-2) * x
    assert "herbie" not in OptimisationOptions().rule_groups
    options = OptimisationOptions(rule_groups=("herbie",))
    assert optimise(expression, options).expression == x
    preserved = optimise(
        expression,
        OptimisationOptions(rule_groups=("herbie",), preserve_floating_point_evaluation=True),
    ).expression
    assert preserved == expression
    invalid = num(0) * (num(1) / x)
    result = optimise(invalid, options).expression
    # At a nonzero input the value is retained; the division cannot be erased.
    assert evaluate(result, {"herbie-x": 2.0}) == 0
    assert result != num(0)
