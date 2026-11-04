# Negative controls: each breaks the passes on purpose, and the tests must notice.
import pytest
import test_pipeline
import test_random

from uflx_spec_interpreter import ScalarType, action, form, function, multiply, num
from uflx_spec_interpreter._src.algorithms import substitution
from uflx_spec_interpreter._src.algorithms.rewriting import _zero_like
from uflx_spec_interpreter._src.passes import actions
from uflx_spec_interpreter._src.rules import binding, differentiation
from uflx_toy.spaces import space


def test_without_freshening_values_change(monkeypatch):
    monkeypatch.setattr(substitution, "_rename", lambda node, name, avoid: name)
    assert test_random.main(2000)["value"] > 0


def test_sequential_substitution_captures(monkeypatch):
    # Binding the members of a tuple entry one at a time captures a later member's function.
    y0, y1 = (
        function(space("V", scalar_type=ScalarType.real), "y0"),
        function(space("V", scalar_type=ScalarType.real), "y1"),
    )
    e = action(form(multiply(y0, y1), ((y0, y1),)), 0, (y1, num(2)))
    assert actions.apply_actions(e) == multiply(y1, num(2))

    def sequential(e, mapping):
        for name, value in mapping.items():
            e = substitution._substitute(e, {name: value})
        return e

    monkeypatch.setattr(binding, "_substitute", sequential)
    assert actions.apply_actions(e) == multiply(num(2), num(2))


def test_a_zero_without_argument_spaces_is_ill_typed(monkeypatch):
    # The values do not change, so only the check of the argument spaces notices.
    def without_spaces(f, spaces, *others):
        return _zero_like(f, (), *others)

    monkeypatch.setattr(differentiation, "_zero_like", without_spaces)
    with pytest.raises(ValueError):
        test_pipeline.main(1000)
