# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Regression examples in the order of the revised language chapters."""

from __future__ import annotations

from collections import namedtuple

import pytest

from uflx._src.nodes import Base, Function, Index, Indexed
from uflx.algorithms import replace
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import conjugate_space, dual_space, form, function
from uflx.nodes import EuclideanSpace, Float, Integer, ScalarType
from uflx.tensors import component, component_tensor, index_sum, indexed, tensor

from .spaces import space


class Payload(Base):
    identifier = "test.Payload"
    payload: object


@pytest.mark.parametrize("payload", [None, [], {}, 1j, namedtuple("Pair", "a")(1)])
def test_construction_arguments_are_curated(payload):
    with pytest.raises(TypeError):
        Payload((payload,))


@pytest.mark.parametrize("payload", [float("nan"), float("inf"), -float("inf")])
def test_nonfinite_arguments_are_rejected_even_in_labels(payload):
    with pytest.raises(ValueError):
        Index((payload,))


def test_constructor_arguments_are_positional_and_typed():
    with pytest.raises(TypeError):
        Integer(value=1)
    assert Float(-0.0) != Float(0.0)
    assert Payload(1) != Payload(1.0)
    with pytest.raises(TypeError):
        tensor([Integer(1)])


def test_empty_index_factories_and_nonempty_nodes():
    f = Integer(2)
    assert indexed(f, ()) is component_tensor(f, ()) is index_sum(f, ()) is f
    A = tensor((f, f))
    assert component(A, ()) is A
    with pytest.raises(IndexError):
        indexed(A, ())
    with pytest.raises(ValueError):
        component_tensor(A, ())
    with pytest.raises(ValueError):
        Indexed(f, ())
    with pytest.raises(TypeError):
        indexed(A, [0])


def test_calculus_domain_is_the_result_domain():
    P = EuclideanSpace(2)
    X = coordinate(P)
    e = integral(X, P)
    assert e.domain is None and e.integration_domain is P
    assert gradient(Integer(1), P).domain is P
    with pytest.raises(ValueError):
        gradient(X, EuclideanSpace(3))
    with pytest.raises(ValueError):
        X[0] + coordinate(EuclideanSpace(3))[0]


def test_space_factories_and_substitution_keep_canonical_forms():
    V = space("V")
    R = space("R", scalar_type=ScalarType.real)
    assert conjugate_space(conjugate_space(V)) is V
    assert dual_space(dual_space(V)) is V
    assert dual_space(dual_space((V, V))) == (V, V)
    assert replace(conjugate_space(V), {V: R}) is R
    u = function(V, "u")
    assert replace(u, {V: R}) is function(R, "u")


def test_replacement_is_simultaneous_and_dictionary_order_independent():
    V = space("R", scalar_type=ScalarType.real)
    u, v, w = (function(V, name) for name in ("u", "v", "w"))
    assert replace(u, {u: v, v: w}) is v
    assert replace(u + v, {u + v: w, u: Integer(1)}) is w
    e = form(u + w, (v,))
    a = replace(e, {u: v, w: Integer(1)})
    b = replace(e, {w: Integer(1), u: v})
    assert a is b
    assert isinstance(a.arguments[0], Function) and a.arguments[0] is not v
