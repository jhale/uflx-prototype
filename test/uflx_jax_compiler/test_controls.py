"""Negative controls deliberately break a pass and check that its tests detect the error."""

import pytest

from uflx._src.algorithms import substitution
from uflx._src.algorithms.rewriting import zero_like
from uflx._src.rules import differentiation

from .test_pipeline import main as pipeline_trials
from .test_random import main as random_trials


def test_without_freshening_values_change(monkeypatch):
    monkeypatch.setattr(substitution, "_rename", lambda node, name, avoid: name)
    assert random_trials(2000)["value"] > 0


def test_a_zero_without_argument_spaces_is_ill_typed(monkeypatch):
    # The values do not change, so only the check of the argument spaces notices.
    def without_spaces(f, spaces, *others):
        return zero_like(f, (), *others)

    monkeypatch.setattr(differentiation, "zero_like", without_spaces)
    with pytest.raises(ValueError):
        pipeline_trials(1000)
