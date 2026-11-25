# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import multiprocessing
from concurrent.futures import ProcessPoolExecutor

import basix

from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space, push_forward
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space


def test_pickled_poisson_form_in_spawned_process():
    triangle = basix.CellType.triangle
    T = topology(triangle)
    p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
    V = local_space(T, p1)
    x = function(local_space(T, p1, shape=(T.dimension,)), "x")
    mesh = create_mesh(x, euclidean_space(T.dimension))

    u, v = function(V, "u"), function(V, "v")
    gu, gv = (gradient(push_forward(f, mesh, V.law), mesh) for f in (u, v))
    a_ref = lower(integral(gv @ gu, mesh))
    rule = basix.make_quadrature(triangle, 0)
    a = form(apply_quadrature(a_ref, *rule), (v, u))

    with ProcessPoolExecutor(
        max_workers=1, mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        assert pool.submit(getattr, a, "digest").result() == a.digest
