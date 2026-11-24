# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import numpy as np

from uflx._src.passes import applications
from uflx._src.passes.applications import applications_expanded
from uflx.arithmetic import add, conjugate, multiply, num
from uflx.calculus import directional_derivative
from uflx.forms import apply, dual_space, form, function, partial
from uflx.tensors import component_tensor, index, indexed, tensor
from uflx_jax_compiler import trace

from ..testutil import ill_typed
from ..uflx.spaces import space

rng = np.random.default_rng(1)
V, Vd = space("V"), dual_space(space("V"))
u, v, w, z, h = (function(V, n) for n in "uvwzh")
l = function(Vd, "l")
i, j = index("i"), index("j")
env = {f: complex(*rng.normal(size=2)) for f in (u, v, w, z, h)}
cl = complex(*rng.normal(size=2))
env[l] = lambda x: cl * x
args = tuple(complex(*rng.normal(size=2)) for _ in range(3))


def check(name, e, ienv=None):
    ienv = ienv or {}
    out = applications.expand_applications(e)
    n = len(e.argument_spaces)
    before = trace(e, *env, *ienv)(*env.values(), *ienv.values(), *args[:n])
    after = trace(out, *env, *ienv)(*env.values(), *ienv.values(), *args[:n])
    assert np.allclose(before, after, atol=1e-6), name
    assert applications_expanded(out), name
    assert not (ill_typed(e) or ill_typed(out)), (name, ill_typed(e) or ill_typed(out))
    return out


def test_cases():
    a = multiply(u, conjugate(v))
    F = form(a, (conjugate(v), u))
    check("bilinear form, action at 1 then 0", apply(F, (z, w)))
    check("action at 0 leaves a linear form", partial(F, {conjugate(v): z}))
    check(
        "sum and product of forms",
        form(apply(add(multiply(num(3), F), F), (conjugate(v), w)), (conjugate(v),)),
    )
    check(
        "product of two linear forms",
        form(
            apply(
                multiply(form(multiply(v, num(2)), (v,)), form(conjugate(u), (conjugate(u),))),
                (v, w),
            ),
            (v,),
        ),
    )
    check("conjugate of a form", form(apply(conjugate(F), (z, conjugate(u))), (conjugate(u),)))
    check("pairing remains", partial(form(multiply(apply(l, (v,)), u), (v, u)), {v: w}))

    # The family index i of g must not be captured by the ComponentTensor that binds i.
    c = tensor((num(1), num(2)))
    g = multiply(indexed(c, (i,)), w)
    G = component_tensor(
        multiply(indexed(tensor((num(5), num(7))), (i,)), form(multiply(v, num(1)), (v,))), (i,)
    )
    for k in range(2):
        check(f"capture, family index i = {k}", apply(G, (g,)), {i: k})

    # The argument contains the variable of the derivative: expect h * u, not 2 h u.
    J = form(directional_derivative(multiply(u, v), u, u, h), (v,))
    out = check("action on a form with a derivative of u at u", apply(J, (u,)))
    assert np.isclose(trace(out, *env)(*env.values()), env[h] * env[u])
