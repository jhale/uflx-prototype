# Negative controls: each breaks the passes on purpose, and the tests must notice.
import pytest
import test_pipeline
import test_random

from uflx import ScalarType, apply, form, function, multiply, num
from uflx._src.algorithms import form_binding, substitution
from uflx._src.algorithms.rewriting import zero_like
from uflx._src.passes import applications
from uflx._src.rules import differentiation
from uflx_basix.spaces import space


def test_without_freshening_values_change(monkeypatch):
    monkeypatch.setattr(substitution, "_rename", lambda node, name, avoid: name)
    assert test_random.main(2000)["value"] > 0


def test_sequential_substitution_captures(monkeypatch):
    # Binding the members of a tuple entry one at a time captures a later member's function.
    y0, y1 = (
        function(space("V", scalar_type=ScalarType.real), "y0"),
        function(space("V", scalar_type=ScalarType.real), "y1"),
    )
    e = apply(form(multiply(y0, y1), ((y0, y1),)), ((y1, num(2)),))
    assert applications.expand_applications(e) == multiply(y1, num(2))

    def sequential(e, mapping):
        for name, value in mapping.items():
            e = substitution.substitute(e, {name: value})
        return e

    monkeypatch.setattr(form_binding, "substitute", sequential)
    assert applications.expand_applications(e) == multiply(num(2), num(2))


def test_a_zero_without_argument_spaces_is_ill_typed(monkeypatch):
    # The values do not change, so only the check of the argument spaces notices.
    def without_spaces(f, spaces, *others):
        return zero_like(f, (), *others)

    monkeypatch.setattr(differentiation, "zero_like", without_spaces)
    with pytest.raises(ValueError):
        test_pipeline.main(1000)
