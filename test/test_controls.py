# Negative controls: each breaks the passes on purpose, and the tests must notice.
import test_random

from uflx_spec_interpreter import substitution
from uflx_spec_interpreter.nodes import Action, Fn, Form, Num, Prod, Space
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
    y0, y1 = Fn(Space("V"), "y0"), Fn(Space("V"), "y1")
    e = Action(Form(Prod(y0, y1), ((y0, y1),)), 0, (y1, Num(2)))
    assert actions.apply_actions(e) == Prod(y1, Num(2))

    def sequential(e, mapping):
        for name, value in mapping.items():
            e = substitution.substitute(e, {name: value})
        return e

    monkeypatch.setattr(actions, "substitute", sequential)
    assert actions.apply_actions(e) == Prod(Num(2), Num(2))
