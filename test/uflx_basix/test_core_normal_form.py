# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from uflx._src.meshes import cell_map
from uflx._src.passes.contracts import in_core_normal_form
from uflx.arithmetic import add, conjugate, multiply, num
from uflx.calculus import coordinate, directional_derivative, gradient, integral
from uflx.forms import apply, dual_space, form, function
from uflx.geometry import pull_back

from ..uflx.library import runtime_array
from ..uflx.spaces import space
from .integrals import D, K, u

V = space("V")
v, w = function(V, "v"), function(V, "w")
l = function(dual_space(V), "l")
cells = cell_map(D, 0)


def test_the_core_normal_form_allows_the_classes_of_its_table():
    for e in (
        num(1),
        coordinate(K),
        gradient(u, D),
        gradient(gradient(u, D), D),
        integral(num(1), K),
        pull_back(u, cells),
        form(multiply(v, w), (v,)),
        apply(l, (conjugate(w),)),
        runtime_array("q", (2,)),
    ):
        assert in_core_normal_form(e), e


def test_the_core_normal_form_rejects_what_its_table_leaves_out():
    for e in (
        gradient(coordinate(K), K),
        gradient(multiply(u, u), D),
        integral(u, D),
        pull_back(multiply(u, u), cells),
        # A cell of the dimension of the domain of the gradient, so the pass rewrites it.
        pull_back(gradient(u, D), cells),
        directional_derivative(multiply(v, v), v, v, w),
        form(directional_derivative(multiply(v, v), v, v, w), (w,)),
        apply(form(multiply(v, w), (v,)), (w,)),
        add(num(1), integral(u, D)),
    ):
        assert not in_core_normal_form(e), e
