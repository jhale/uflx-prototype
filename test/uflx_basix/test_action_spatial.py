# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Spatial operators transform a form's output without transforming its arguments."""

from __future__ import annotations

from uflx._src.nodes import Apply, SideMap
from uflx._src.passes.contracts import core_normal_form
from uflx._src.passes.pullbacks import pull_backs_expanded
from uflx.arithmetic import num
from uflx.calculus import coordinate
from uflx.forms import apply, form, function
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace, ScalarType
from uflx.passes import expand_applications, expand_pull_backs
from uflx.tensors import index, tensor

from ..uflx.spaces import space
from ..uflx.test_action_spatial import OperatorSpace
from .cells import reference_simplex


def setup():
    P, K = EuclideanSpace(1), reference_simplex(1)
    x, t = coordinate(P)[0], coordinate(K)[0]
    v = function(space("spatial-argument", P, scalar_type=ScalarType.real), "v")
    A = form(x * v, (v,))
    psi = domain_map(K, tensor((num(1) + num(2) * t,)), P)
    return P, K, x, v, A, psi


def test_side_pullback_transforms_output_before_deferral_and_keeps_tuple_family():
    import basix

    from uflx_basix.cells import cell

    from .meshes import side_map, square

    D, E, _, _ = square(cell(basix.CellType.triangle))
    sm = side_map(D, E)
    assert isinstance(sm, SideMap)
    V = space("tuple-input", D, scalar_type=ScalarType.real)
    A = function(OperatorSpace(D, ((V, V),)), "tuple-operator")
    u, v = function(V, "u"), function(V, "v")
    k = index("argument-family")
    g = (tensor((u, v))[k], v)
    original = pull_back(apply(A, (g,)), sm)
    assert not pull_backs_expanded(original)
    out = expand_pull_backs(original)
    assert isinstance(out, Apply)
    assert out.arguments[0] == g
    assert out.form is pull_back(A, sm)
    assert out.free_indices == original.free_indices
    assert pull_backs_expanded(out) and not core_normal_form(out)
    assert expand_applications(out) is out


def test_pullback_of_bound_form_keeps_binding_entries_in_the_original_space():
    _, _, _, v, A, psi = setup()
    out = expand_pull_backs(pull_back(A, psi))
    assert out.arguments == (v,)
    assert out.argument_spaces == A.argument_spaces
