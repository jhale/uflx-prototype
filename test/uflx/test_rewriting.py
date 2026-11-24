# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Replacement processing and recipe termination."""

import pytest

from uflx._src.algorithms.rewriting import rewrite
from uflx._src.algorithms.traversal import map_construction, rebuild
from uflx._src.passes import lowering, spatial
from uflx.algorithms import bottom_up, replace
from uflx.arithmetic import num
from uflx.calculus import gradient
from uflx.forms import form, function
from uflx.nodes import EuclideanSpace
from uflx.passes import apply_gradients, lower
from uflx.tensors import tensor

from .spaces import space


def test_mapping_nested_operands_reuses_unchanged_tuples():
    a, b = num(1), num(2)
    shared = (a, a)
    arguments = (shared, (b, shared))
    assert map_construction(arguments, lambda node: node) is arguments

    visits = []

    def replace(node):
        visits.append(node)
        return a if node is b else node

    result = map_construction(arguments, replace)
    assert result == (shared, (a, shared))
    assert result[0] is result[1][1] is shared
    assert visits == [a, a, b, a, a]


def test_deep_shared_graph_lowering_and_scoped_replacement():
    V = space("deep", None, scalar_type="real")
    x, y = function(V, "x"), function(V, "y")
    original, expected = x, y
    for _ in range(1000):
        original = original / original - original
        expected = expected / expected - expected
    assert lower(original) is original
    assert bottom_up(original, lambda e: y if e is x else e) is expected
    assert replace(original, {x: y}) is expected

    replaced = replace(form(original + y, (y,)), {x: y})
    (fresh,) = replaced.arguments
    assert fresh is not y
    assert replaced.expression is expected + fresh


def test_deep_replacement_chain_is_resolved_in_order():
    visited = []

    def rule(e):
        visited.append(e.value)
        return num(e.value + 1) if e.value < 1500 else e

    assert rewrite(num(0), rule) is num(1500)
    assert visited == list(range(1501))


def test_mapping_deep_tuple_keeps_unchanged_branches():
    a, b = num(1), num(2)
    data = a
    for _ in range(1500):
        data = (data,)
    assert map_construction(data, lambda e: e) is data
    result = map_construction(data, lambda e: b)
    for _ in range(1500):
        (result,) = result
    assert result is b


def test_replacement_operands_are_processed_and_shared_nodes_reused():
    calls = []

    def rule(e):
        calls.append(e)
        return num(2) if e == num(1) else e

    out = rewrite(tensor((num(1), num(1))), rule)
    assert out == tensor((num(2), num(2)))
    assert calls.count(num(1)) == calls.count(num(2)) == 1
    assert rebuild(out, entries=out.entries) is out


def test_fixed_and_dynamic_visits_on_shared_branches():
    a, b, c, d = map(num, (1, 2, 3, 4))
    shared, later = a + b, c + a
    root = tensor((shared, later, a, shared))
    calls = []

    def rule(node):
        calls.append(node)
        return {b: later, c: d}.get(node, node)

    # A fixed pass uses the replacement as supplied, even if it is a later sibling.
    fixed = tensor((a + later, d + a, a, a + later))
    assert bottom_up(root, rule) is fixed
    assert calls == [a, b, a + later, c, d + a, fixed]

    # A dynamic pass finishes that sibling immediately and reuses its result.
    resolved = a + (d + a)
    dynamic = tensor((resolved, d + a, a, resolved))
    calls.clear()
    assert rewrite(root, rule) is dynamic
    assert calls == [a, b, c, d, d + a, resolved, dynamic]


def test_replacement_cannot_request_an_original_ancestor():
    a, b = num(1), num(2)
    root = a + b
    with pytest.raises(ValueError, match="rewrite cycle") as exc:
        rewrite(root, lambda node: root if node is b else node)
    assert exc.value.args[1] is root


def test_replacement_cycles_raise():
    with pytest.raises(ValueError, match="rewrite cycle"):
        rewrite(num(1), lambda e: num(2) if e == num(1) else num(1))


def test_retained_gradient_is_not_reconstructed_on_each_visit():
    P = EuclideanSpace(2)
    u = function(space("field", P, value_shape=(2,)), "u")
    for e in (gradient(u, P), gradient(gradient(u, P), P)):
        assert apply_gradients(e) is e
        assert rewrite(e, lambda n: n) is e


def test_spatial_round_trip_is_a_cycle_not_a_fixed_point(monkeypatch):
    monkeypatch.setattr(spatial, "apply_gradients", lambda e: num(2))
    monkeypatch.setattr(spatial, "expand_pull_backs", lambda e: num(1))
    with pytest.raises(ValueError, match="spatial lowering cycle"):
        lowering.lower_spatial(num(1))


def test_spatial_cycle_across_rounds_raises(monkeypatch):
    monkeypatch.setattr(spatial, "apply_gradients", lambda e: num(2) if e == num(1) else num(1))
    monkeypatch.setattr(spatial, "expand_pull_backs", lambda e: e)
    with pytest.raises(ValueError, match="spatial lowering cycle"):
        lowering.lower_spatial(num(1))
