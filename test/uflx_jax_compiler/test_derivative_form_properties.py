# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import random
import sys

import numpy as np

import uflx_jax_compiler
from uflx._src.passes import applications, derivatives
from uflx.arithmetic import absolute, add, conjugate, multiply, num, real_part
from uflx.calculus import directional_derivative
from uflx.forms import apply, dual_space, form, function, partial
from uflx.nodes import ScalarType
from uflx_extensions.complex_differentiation import complex_derivative_parts

from ..testutil import assert_clean, check_types, ill_typed
from ..uflx.spaces import space
from .reference import reference_trace

C, R = space("C", None), space("R", None, scalar_type=ScalarType.real)


# A load in each dual space, linear in its argument like every action.
def load(V):
    return function(dual_space(V), "l")


LOAD_VALUE = 0.7 - 0.4j


def body(rnd, u, names, holomorphic, depth):
    if depth == 0 or rnd.random() < 0.3:
        return rnd.choice([u, num(2), *names])
    a, b = body(rnd, u, names, holomorphic, depth - 1), body(rnd, u, names, holomorphic, depth - 1)
    l = load(u.space)
    ops = [lambda: add(a, b), lambda: multiply(a, b), lambda: apply(l, (a,))]
    if not holomorphic:
        ops += [
            lambda: conjugate(a),
            lambda: absolute(add(num(2), a)),
            lambda: real_part(a),
            lambda: apply(l, (conjugate(a),)),
        ]
    return rnd.choice(ops)()


def main(trials):
    bad = 0
    rows = {"real": 0, "holomorphic": 0, "general": 0}
    typed = {}
    for seed in range(trials):
        rnd = random.Random(seed)
        space = rnd.choice([C, R])
        u, v, w, z = (function(space, n) for n in "uvwz")
        holomorphic = rnd.random() < 0.5
        test = conjugate(v) if (space is C and rnd.random() < 0.5) else v
        F = form(
            multiply(
                multiply(v if test == v else conjugate(v), u), body(rnd, u, [], holomorphic, 3)
            ),
            (test,),
        )
        entry = conjugate(w) if rnd.random() < 0.5 else w
        e = form(
            directional_derivative(F, u, u, w)
            if u.space.scalar_type == "real"
            else complex_derivative_parts(F, u, u, w)[int(entry != w)],
            (entry,),
        )
        if rnd.random() < 0.4:
            # An action on the new argument, the derivative in a fixed direction.
            e = partial(e, {entry: rnd.choice([z, add(u, z)])})
        rows["real" if space is R else ("holomorphic" if holomorphic else "general")] += 1
        out = derivatives.apply_derivatives(applications.expand_applications(e))
        check_types(typed, e, out)
        rng = np.random.default_rng(seed)
        scalar = (
            (lambda: complex(*rng.normal(size=2))) if space is C else (lambda: float(rng.normal()))
        )
        env = {f: 0.5 * scalar() for f in (u, v, w, z)}
        value = LOAD_VALUE if space is C else LOAD_VALUE.real
        env[load(space)] = lambda x, value=value: value * x
        vals = tuple(0.5 * scalar() for _ in range(len(e.argument_spaces)))
        before, after = (
            reference_trace(e, *env)(*env.values(), *vals),
            uflx_jax_compiler.trace(out, *env)(*env.values(), *vals),
        )
        if not np.allclose(before, after, rtol=1e-5, atol=1e-6):
            bad += 1
            if bad <= 3:
                print("value changed, seed", seed, space.name, holomorphic, entry, before, after)
    print(f"{trials} derivative forms: {bad} values changed; cases {rows}")
    return dict(value=bad) | typed


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)


def test_projected_derivative_forms():
    assert_clean(main(400))


def test_pairing_with_the_conjugate():
    # <l, u> u is written with conjugate(u), so it is not holomorphic in u, and J- is not zero.
    u, w = function(C, "u"), function(C, "w")
    l = load(C)
    F = multiply(apply(l, (conjugate(u),)), u)
    env = {u: 0.3 + 0.2j, w: 0.1 - 0.5j, l: lambda x: LOAD_VALUE * x}
    for entry in (w, conjugate(w)):
        e = form(
            directional_derivative(F, u, u, w)
            if u.space.scalar_type == "real"
            else complex_derivative_parts(F, u, u, w)[int(entry != w)],
            (entry,),
        )
        out = derivatives.apply_derivatives(applications.expand_applications(e))
        assert not (ill_typed(e) or ill_typed(out))
        before, after = (
            reference_trace(e, *env)(*env.values(), 0.4 + 0.9j),
            uflx_jax_compiler.trace(out, *env)(*env.values(), 0.4 + 0.9j),
        )
        assert np.isclose(before, after) and abs(after) > 0.1
