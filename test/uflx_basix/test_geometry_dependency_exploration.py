# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Geometry exposure probes against finite differences and independent field data."""

import pytest

from uflx._src.algorithms.substitution import occurs, occurs_in_metadata
from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import form, function
from uflx.passes import apply_derivatives

from ..uflx.spaces import space
from .shape_derivatives import D, functionals, v0, v1, x0, x1


def test_geometry_dependence_is_detected_even_with_value_dependence():
    f = functionals()["area"] * x0
    assert occurs_in_metadata(x0, f)


def test_unknown_geometry_dependent_space_is_rejected():
    u = function(space("unknown-physical", D, scalar_type="real"), "u")
    with pytest.raises(ValueError, match="unresolved geometry or space dependence"):
        apply_derivatives(directional_derivative(u, x0, x0, v0))
    # An unused declaration still carries its parameter-space dependency.
    assert occurs(x0, form(num(1), (u,)))


def test_ordinary_coefficient_derivative_preserves_geometry():
    V = space("physical-coefficient", D, scalar_type="real")
    u, w = function(V, "u"), function(V, "w")
    assert not occurs_in_metadata(u, u * u)
    assert apply_derivatives(directional_derivative(u * u, u, u, w)) is w * u + u * w


def test_dependent_binder_spaces_are_explicitly_unsupported():
    u = function(space("dependent-parameter", D, scalar_type="real"), "u")
    with pytest.raises(ValueError, match="argument space depends"):
        form(u, (x0, u))
    with pytest.raises(ValueError, match="variable spaces must be independent"):
        directional_derivative(u, (x0, u), (x0, u), (v0, u))


def test_nested_hidden_geometry_derivative_is_resolved_inside_out():
    inner = directional_derivative(functionals()["area"], x0, x0, v0)
    nested = apply_derivatives(directional_derivative(inner, x1, x1, v1))
    staged = apply_derivatives(
        directional_derivative(
            apply_derivatives(directional_derivative(functionals()["area"], x0, x0, v0)), x1, x1, v1
        )
    )
    assert nested is staged


def test_dependent_inherited_slots_are_also_rejected():
    u = function(space("dependent-inner-parameter", D, scalar_type="real"), "u")
    with pytest.raises(ValueError, match="argument space depends"):
        form(form(u, (u,)), (x0,))
