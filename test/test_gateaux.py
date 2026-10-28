import random
import sys

import numpy as np
from testutil import assert_clean

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    absolute,
    action,
    add,
    conjugate,
    dual_space,
    form,
    function,
    gateaux_derivative,
    multiply,
    num,
    real_part,
)
from uflx_spec_interpreter._src.passes import actions, derivatives, gateaux
from uflx_spec_interpreter._src.static import arity
from uflx_toy.spaces import space

C, R = space("C"), space("R", scalar_type="real")
# A load in each dual space, linear in its argument like every action.
LOADS = {
    C: function(dual_space(space("C")), "l"),
    R: function(dual_space(space("R", scalar_type="real")), "l"),
}
LOAD_VALUE = 0.7 - 0.4j


def body(rnd, u, names, holomorphic, depth):
    if depth == 0 or rnd.random() < 0.3:
        return rnd.choice([u, num(2), *names])
    a, b = body(rnd, u, names, holomorphic, depth - 1), body(rnd, u, names, holomorphic, depth - 1)
    l = LOADS[u.space]
    ops = [lambda: add(a, b), lambda: multiply(a, b), lambda: action(l, 0, a)]
    if not holomorphic:
        ops += [
            lambda: conjugate(a),
            lambda: absolute(add(num(2), a)),
            lambda: real_part(a),
            lambda: action(l, 0, conjugate(a)),
        ]
    return rnd.choice(ops)()


def main(trials):
    bad, rejected = 0, 0
    rows = {"real": 0, "holomorphic": 0, "general": 0}
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
        e = gateaux_derivative(F, u, entry, u)
        if rnd.random() < 0.4:
            # An action on the new argument, the derivative in a fixed direction.
            e = action(e, 1, rnd.choice([z, add(u, z)]))
        rows["real" if space is R else ("holomorphic" if holomorphic else "general")] += 1
        out0 = gateaux.apply_gateaux_derivatives(e)
        rejected += not gateaux.in_n1(e)
        bad += not gateaux.in_n1(out0)
        out = derivatives.apply_derivatives(actions.apply_actions(out0))
        rng = np.random.default_rng(seed)
        scalar = (
            (lambda: complex(*rng.normal(size=2))) if space is C else (lambda: float(rng.normal()))
        )
        env = {f: 0.5 * scalar() for f in (u, v, w, z)}
        load = LOAD_VALUE if space is C else LOAD_VALUE.real
        env[LOADS[space]] = lambda x, load=load: load * x
        vals = tuple(0.5 * scalar() for _ in range(arity(e)))
        before, after = evaluate(e, env)(vals), evaluate(out, env)(vals)
        if not np.allclose(before, after, rtol=1e-5, atol=1e-6):
            bad += 1
            if bad <= 3:
                print("value changed, seed", seed, space.name, holomorphic, entry, before, after)
    print(
        f"{trials} Gateaux derivatives: {bad} values changed or outside N1; "
        f"N1 rejected {rejected} of {trials} inputs; "
        f"cases {rows}"
    )
    return dict(value=bad, lazy=trials, rejected=rejected)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)


def test_gateaux():
    assert_clean(main(400))


def test_pairing_with_the_conjugate():
    # <l, u> u is written with conjugate(u), so it is not holomorphic in u, and J- is not zero.
    u, w = function(C, "u"), function(C, "w")
    l = LOADS[C]
    F = multiply(action(l, 0, conjugate(u)), u)
    env = {u: 0.3 + 0.2j, w: 0.1 - 0.5j, l: lambda x: LOAD_VALUE * x}
    assert not gateaux.holomorphic(F, u)
    for entry in (w, conjugate(w)):
        e = gateaux_derivative(F, u, entry, u)
        out = derivatives.apply_derivatives(
            actions.apply_actions(gateaux.apply_gateaux_derivatives(e))
        )
        before, after = evaluate(e, env)((0.4 + 0.9j,)), evaluate(out, env)((0.4 + 0.9j,))
        assert np.isclose(before, after) and abs(after) > 0.1
