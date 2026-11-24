# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import basix

from uflx._src.nodes import Conjugate
from uflx_basix.cells import cell

from .legacy import (
    Coefficient,
    Constant,
    FacetNormal,
    FunctionSpace,
    Measure,
    Mesh,
    TestFunction,
    TrialFunction,
    VectorFunctionSpace,
    div,
    grad,
    inner,
)
from .meshes import grouped_topology, vertex_mesh

K, I = cell(basix.CellType.triangle), cell(basix.CellType.interval)
T = grouped_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
BOUNDARY = grouped_topology((I,), (((0, 1), (1, 2), (2, 3), (0, 3)),), ((("inside", T, 0),),))
LOWER = grouped_topology((K,), (((0, 1, 2),),), ((("inside", T, 0),),))
LEFT = grouped_topology((I,), (((0, 3),),), ((("inside", T, 0),),))
D, GEOMETRY = vertex_mesh(T, [(0, 0), (1, 0), (1.1, 0.9), (-0.1, 1)])


def demo():
    # The user script of EXAMPLES.md, unchanged.
    mesh = Mesh(D, BOUNDARY)
    V = FunctionSpace(mesh, "Lagrange", 1)
    W = VectorFunctionSpace(mesh, "Lagrange", 1)
    u, v = TrialFunction(V), TestFunction(V)
    f, q = Coefficient(V, "f"), Coefficient(W, "q")
    kappa = Constant("kappa")
    n = FacetNormal(mesh)
    dx = Measure("dx", domain=mesh, subdomain_data={1: LOWER})
    ds = Measure("ds", domain=mesh, subdomain_data={1: LEFT})
    a = kappa * inner(grad(u), grad(v)) * dx
    L = inner(f, v) * dx(1) - inner(div(q), v) * dx + inner(inner(q, n), v) * ds(1)
    return a, L


def test_the_arguments_of_a_classic_form_are_the_functions_on_the_topology():
    a, _ = demo()
    v, u = a.arguments
    assert isinstance(v, Conjugate) and v.operand.domain == T and u.domain == T
