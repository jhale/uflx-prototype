# Negative controls: each breaks the passes on purpose, and the tests must notice.
import pkgutil
from importlib import import_module

import pytest
import test_pipeline
import test_random

import uflx_spec_interpreter._src as core
from uflx_spec_interpreter import action, form, function, multiply, num
from uflx_spec_interpreter._src import static, substitution
from uflx_spec_interpreter._src.passes import actions, derivatives
from uflx_toy.spaces import space


def clear_caches():
    # The passes cache their results, which a broken pass must not reuse or leave behind.
    for info in pkgutil.walk_packages(core.__path__, core.__name__ + "."):
        for x in vars(import_module(info.name)).values():
            if hasattr(x, "cache_clear"):
                x.cache_clear()


@pytest.fixture
def patch(monkeypatch):
    def set_attribute(module, name, value):
        monkeypatch.setattr(module, name, value)
        clear_caches()

    yield set_attribute
    monkeypatch.undo()
    clear_caches()


def test_without_freshening_values_change(patch):
    patch(substitution, "rename", lambda node, name, avoid: name)
    assert test_random.main(2000)["value"] > 0


def test_sequential_substitution_captures(patch):
    # Binding the members of a tuple entry one at a time captures a later member's function.
    y0, y1 = function(space("V"), "y0"), function(space("V"), "y1")
    e = action(form(multiply(y0, y1), ((y0, y1),)), 0, (y1, num(2)))
    assert actions.apply_actions(e) == multiply(y1, num(2))

    def sequential(e, mapping):
        for name, value in mapping.items():
            e = substitution.substitute(e, {name: value})
        return e

    patch(actions, "substitute", sequential)
    assert actions.apply_actions(e) == multiply(num(2), num(2))


def test_a_zero_without_argument_spaces_is_ill_typed(patch):
    # The values do not change, so only the check of the argument spaces notices.
    def without_spaces(f, spaces, *others):
        return static.zero_like(f, (), *others)

    patch(derivatives, "zero_like", without_spaces)
    counts = test_pipeline.main(1000)
    assert counts["typed"] > 0 and counts["value"] == 0
