# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Diagnostic contracts are read-only checks, independent of lowering recipes."""

from uflx._src.nodes import DirectionalDerivative
from uflx._src.passes.contracts import (
    core_normal_form,
    derivatives_expanded,
    in_core_normal_form,
    reference_integration,
)
from uflx.calculus import directional_derivative, gradient
from uflx.forms import function
from uflx.nodes import EuclideanSpace
from uflx.passes import apply_derivatives
from uflx.tensors import tensor

from .spaces import space


def test_reports_shared_nodes_once_and_does_not_lower():
    u, h = (function(space("contract", None), name) for name in ("u", "h"))
    d = directional_derivative(u * u, u, u, h)
    e = tensor((d, d))
    assert derivatives_expanded(e) == ((d, "directional derivative remains"),)
    assert isinstance(e.entries[0], DirectionalDerivative)
    assert not derivatives_expanded(apply_derivatives(e))


def test_reports_multiple_failures_in_stable_order():
    u, h = (function(space("contract-order", None), name) for name in ("u", "h"))
    first = directional_derivative(u, u, u, h)
    second = directional_derivative(u * u, u, u, h)
    e = tensor((first, second, first))
    assert tuple(n for n, _ in derivatives_expanded(e)) == (first, second)
    assert derivatives_expanded(e) == derivatives_expanded(e)


def test_nonexpanded_gradient_is_allowed_at_earlier_boundaries():
    P = EuclideanSpace(2)
    u = function(space("gradient-contract", P), "u")
    e = gradient(u * u, P)
    assert not derivatives_expanded(e)
    assert not reference_integration(e)
    assert core_normal_form(e) == ((e, "Gradient is outside core normal form"),)
    assert bool(core_normal_form(e)) == (not in_core_normal_form(e))
    assert not core_normal_form(gradient(u, P))
