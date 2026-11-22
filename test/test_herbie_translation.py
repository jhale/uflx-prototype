"""Check source provenance, conservative translation and executable Herbie identities."""

import json
import random
import runpy
from fractions import Fraction
from pathlib import Path
from types import SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[1]
translator = runpy.run_path(str(ROOT / "tools/translate_herbie.py"))


def test_reader_ignores_line_and_datum_comments():
    source = """
    ; (define-rules arithmetic [bad (+ x 1) 0])
    (define-rules arithmetic
      #;[bad (- x x) 1]
      [good (+ x 0) x])
    """
    assert list(translator["catalogue"](source)) == [
        ("arithmetic", "good", ("+", "x", "0"), "x", ())
    ]


def test_generated_files_match_pinned_source():
    vendor = ROOT / "third_party/herbie"
    code, report = translator["generate"](
        (vendor / "rules.rkt").read_text(), (vendor / "LICENSE.md").read_text()
    )
    assert code == (ROOT / "src/uflx/_src/optimisation/herbie.py").read_text()
    assert report == json.loads((vendor / "translation.json").read_text())
    assert len(report["rules"]) == 479
    assert sum(r["status"] == "translated" for r in report["rules"]) == 75
    assert "sum-to-mult" not in {r["name"] for r in report["rules"]}  # #; commented upstream


def exact(expression, env):
    """Independent evaluation of the upstream arithmetic subset."""
    if isinstance(expression, str):
        return env[expression] if expression in env else Fraction(expression)
    op, *operands = expression
    args = [exact(x, env) for x in operands]
    if op == "neg":
        return -args[0]
    a, b = args
    return {
        "+": lambda: a + b,
        "-": lambda: a - b,
        "*": lambda: a * b,
        "/": lambda: a / b,
        "pow": lambda: a ** int(b),
    }[op]()


def test_every_translated_rule_matches_exact_upstream_arithmetic():
    source = (ROOT / "third_party/herbie/rules.rkt").read_text()
    included = {
        r["name"]
        for r in json.loads((ROOT / "third_party/herbie/translation.json").read_text())["rules"]
        if r["status"] == "translated"
    }
    rng = random.Random(17)
    for _, name, lhs, rhs, _ in translator["catalogue"](source):
        if name not in included:
            continue
        names = translator["variables"](lhs)
        for sample in range(12):
            env = {
                key: Fraction(rng.randrange(-5, 6), rng.randrange(1, 6)) if sample else Fraction(0)
                for key in sorted(names)
            }
            expected = exact(lhs, env)
            assert expected == exact(rhs, env), name
            namespace = {"Scalar": SimpleNamespace(number=lambda x: x), "BigRat": Fraction, **env}
            for expression in (lhs, rhs):
                assert (
                    eval(translator["translate"](expression), {"__builtins__": {}}, namespace)
                    == expected
                ), name


@pytest.mark.parametrize(
    "expression",
    [("/", "a", "b"), ("/", "1", "0"), ("pow", "a", "b"), ("pow", "a", "1/2"), ("sqrt", "a")],
)
def test_restricted_domains_are_not_silently_translated(expression):
    with pytest.raises(ValueError):
        translator["translate"](expression)


def test_translated_rules_collect_signed_terms_and_guard_cancellation():
    egglog = pytest.importorskip("egglog")
    from uflx._src.optimisation.herbie import herbie_arithmetic, herbie_polynomials
    from uflx._src.optimisation.language import Scalar
    from uflx._src.optimisation.rules import constant_cleanup, defined

    n = lambda i: Scalar.number(egglog.BigRat(i, 1))
    x = Scalar.atom(0)
    rules = herbie_arithmetic | herbie_polynomials | constant_cleanup
    graph = egglog.EGraph()
    expression = n(3) * x + n(-2) * x
    graph.register(expression)
    graph.run(rules * 3)
    graph.check(egglog.eq(expression).to(x))
    cancellation = x + n(-1) * x
    graph = egglog.EGraph()
    graph.register(cancellation)
    graph.run(rules * 3)
    assert str(graph.extract(cancellation)) != str(n(0))
    graph = egglog.EGraph()
    graph.register(cancellation, defined(x))
    graph.run(rules * 3)
    graph.check(egglog.eq(cancellation).to(n(0)))


def test_herbie_group_is_opt_in_and_respects_floating_point_policy():
    pytest.importorskip("egglog")
    from uflx import num
    from uflx._src.optimisation import OptimisationOptions, optimise
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
