"""Replacement processing and recipe termination."""

import pytest

from uflx_spec_interpreter import RD, apply_gradients, function, gradient, num, tensor
from uflx_spec_interpreter._src.passes import lowering
from uflx_spec_interpreter._src.rewriting import _rewrite
from uflx_spec_interpreter._src.traversal import _rebuild
from uflx_toy.spaces import space


def test_replacement_operands_are_processed_and_shared_nodes_reused():
    calls = []

    def rule(e):
        calls.append(e)
        return num(2) if e == num(1) else e

    out = _rewrite(tensor((num(1), num(1))), rule)
    assert out == tensor((num(2), num(2)))
    assert calls.count(num(1)) == calls.count(num(2)) == 1
    assert _rebuild(out, entries=out.entries) is out


def test_replacement_cycles_raise():
    with pytest.raises(ValueError, match="rewrite cycle"):
        _rewrite(num(1), lambda e: num(2) if e == num(1) else num(1))


def test_retained_gradient_is_not_reconstructed_on_each_visit():
    P = RD(2)
    u = function(space("field", domain=P, value_shape=(2,)), "u")
    for e in (gradient(u, P), gradient(gradient(u, P), P)):
        assert apply_gradients(e) is e
        assert _rewrite(e, lambda n: n) is e


def test_spatial_round_trip_is_a_cycle_not_a_fixed_point(monkeypatch):
    monkeypatch.setattr(lowering, "apply_gradients", lambda e: num(2))
    monkeypatch.setattr(lowering, "expand_pull_backs", lambda e: num(1))
    with pytest.raises(ValueError, match="spatial lowering cycle"):
        lowering._lower_spatial(num(1))


def test_spatial_cycle_across_rounds_raises(monkeypatch):
    monkeypatch.setattr(lowering, "apply_gradients", lambda e: num(2) if e == num(1) else num(1))
    monkeypatch.setattr(lowering, "expand_pull_backs", lambda e: e)
    with pytest.raises(ValueError, match="spatial lowering cycle"):
        lowering._lower_spatial(num(1))
