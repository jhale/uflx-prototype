# Negative controls: each breaks the passes on purpose, and the tests must notice.
import test_random

from uflx_spec_interpreter import substitution
from uflx_spec_interpreter.nodes import Action, Form, Function, Num, Product, Space
from uflx_spec_interpreter.passes import actions


def test_without_freshening_values_change(monkeypatch):
    monkeypatch.setattr(substitution, "rename", lambda node, name, avoid: name)
    substitution._substitute.cache_clear()
    try:
        assert test_random.main(2000)["value"] > 0
    finally:
        substitution._substitute.cache_clear()


def test_sequential_substitution_captures(monkeypatch):
    # Binding the members of a tuple entry one at a time captures a later member's function.
    y0, y1 = Function(Space("V"), "y0"), Function(Space("V"), "y1")
    e = Action(Form(Product(y0, y1), ((y0, y1),)), 0, (y1, Num(2)))
    assert actions.apply_actions(e) == Product(y1, Num(2))

    def sequential(e, mapping):
        for name, value in mapping.items():
            e = substitution.substitute(e, {name: value})
        return e

    monkeypatch.setattr(actions, "substitute", sequential)
    assert actions.apply_actions(e) == Product(Num(2), Num(2))
