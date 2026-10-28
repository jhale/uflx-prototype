import random
import sys

import numpy as np
from testutil import assert_clean

from uflx_spec_interpreter.nodes import (
    Abs,
    Action,
    Conj,
    Fn,
    Form,
    Gateaux,
    Num,
    Prod,
    Re,
    Space,
    Sum,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import actions, derivatives, gateaux
from uflx_spec_interpreter.static import arity

C, R = Space("C"), Space("R", complex=False)
# A load in each dual space, linear in its argument like every action.
LOADS = {C: Fn(Space("C", dual=True), "l"), R: Fn(Space("R", dual=True, complex=False), "l")}
LOAD_VALUE = 0.7 - 0.4j


def body(rnd, u, names, holomorphic, depth):
    if depth == 0 or rnd.random() < 0.3:
        return rnd.choice([u, Num(2), *names])
    a, b = body(rnd, u, names, holomorphic, depth - 1), body(rnd, u, names, holomorphic, depth - 1)
    l = LOADS[u.space]
    ops = [lambda: Sum(a, b), lambda: Prod(a, b), lambda: Action(l, 0, a)]
    if not holomorphic:
        ops += [
            lambda: Conj(a),
            lambda: Abs(Sum(Num(2), a)),
            lambda: Re(a),
            lambda: Action(l, 0, Conj(a)),
        ]
    return rnd.choice(ops)()


def main(trials):
    bad, rejected = 0, 0
    rows = {"real": 0, "holomorphic": 0, "general": 0}
    for seed in range(trials):
        rnd = random.Random(seed)
        space = rnd.choice([C, R])
        u, v, w, z = (Fn(space, n) for n in "uvwz")
        holomorphic = rnd.random() < 0.5
        test = Conj(v) if (space is C and rnd.random() < 0.5) else v
        F = Form(
            Prod(Prod(v if test == v else Conj(v), u), body(rnd, u, [], holomorphic, 3)), (test,)
        )
        entry = Conj(w) if rnd.random() < 0.5 else w
        e = Gateaux(F, u, u, entry)
        if rnd.random() < 0.4:
            # An action on the new argument, the derivative in a fixed direction.
            e = Action(e, 1, rnd.choice([z, Sum(u, z)]))
        rows["real" if space is R else ("holomorphic" if holomorphic else "general")] += 1
        out0 = gateaux.apply_gateaux(e)
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
    u, w = Fn(C, "u"), Fn(C, "w")
    l = LOADS[C]
    F = Prod(Action(l, 0, Conj(u)), u)
    env = {u: 0.3 + 0.2j, w: 0.1 - 0.5j, l: lambda x: LOAD_VALUE * x}
    assert not gateaux.holomorphic(F, u)
    for entry in (w, Conj(w)):
        e = Gateaux(F, u, u, entry)
        out = derivatives.apply_derivatives(actions.apply_actions(gateaux.apply_gateaux(e)))
        before, after = evaluate(e, env)((0.4 + 0.9j,)), evaluate(out, env)((0.4 + 0.9j,))
        assert np.isclose(before, after) and abs(after) > 0.1
