"""Negative controls deliberately break a pass and check that its tests detect the error."""

from uflx._src import forms
from uflx._src.algorithms import substitution
from uflx._src.passes import applications
from uflx.arithmetic import multiply, num
from uflx.forms import apply, form, function
from uflx.nodes import ScalarType

from .spaces import space


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

    monkeypatch.setattr(forms, "substitute", sequential)
    assert applications.expand_applications(e) == multiply(num(2), num(2))
