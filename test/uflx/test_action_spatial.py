# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Spatial operators transform a form's output without transforming its arguments."""

from __future__ import annotations

from uflx._src.nodes import AbstractDomain, AbstractFunctionSpace
from uflx._src.passes.applications import applications_expanded
from uflx._src.passes.contracts import core_normal_form
from uflx.calculus import gradient
from uflx.forms import apply, function
from uflx.nodes import EuclideanSpace, ScalarType
from uflx.passes import expand_applications, lower

from .spaces import space


class OperatorSpace(AbstractFunctionSpace):
    """A library-supplied operator with field-valued output and specified slots."""

    identifier = "test.OperatorSpace"
    _domain: AbstractDomain
    _argument_spaces: tuple
    value_shape = ()
    scalar_type = ScalarType.real

    @property
    def domain(self):
        return self._domain

    @property
    def argument_spaces(self):
        return self._argument_spaces


def test_partial_applications_of_transformed_operators_remain_and_are_accepted():
    P = EuclideanSpace(1)
    V = space("operator-input", domain=P, scalar_type=ScalarType.real)
    A = function(OperatorSpace(P, (V, V)), "operator")
    u, v = function(V, "u"), function(V, "v")
    e = apply(gradient(A, P), (v, u))
    assert expand_applications(e) is e
    assert applications_expanded(e)
    assert not core_normal_form(e)
    assert lower(e) is e
