import random
import sys

import numpy as np

from uflx_spec_interpreter import derivatives
from uflx_spec_interpreter import gateaux
from uflx_spec_interpreter import actions
from uflx_spec_interpreter.core import Abs, Action, Conj, Fn, Form, Gateaux, Num, Prod, Re, Space, Sum
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.static import arity
from testutil import assert_clean

C, R = Space("C"), Space("R", complex=False)


def body(rnd, u, names, holomorphic, depth):
    if depth == 0 or rnd.random() < 0.3:
        return rnd.choice([u, Num(2)] + names)
    a, b = body(rnd, u, names, holomorphic, depth - 1), body(rnd, u, names, holomorphic, depth - 1)
    ops = [lambda: Sum(a, b), lambda: Prod(a, b)]
    if not holomorphic:
        ops += [lambda: Conj(a), lambda: Abs(Sum(Num(2), a)), lambda: Re(a)]
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
        F = Form(Prod(Prod(v if test == v else Conj(v), u), body(rnd, u, [], holomorphic, 3)), (test,))
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
        scalar = (lambda: complex(*rng.normal(size=2))) if space is C else (lambda: float(rng.normal()))
        env = {f: 0.5 * scalar() for f in (u, v, w, z)}
        vals = tuple(0.5 * scalar() for _ in range(arity(e)))
        before, after = evaluate(e, env)(vals), evaluate(out, env)(vals)
        if not np.allclose(before, after, rtol=1e-5, atol=1e-6):
            bad += 1
            if bad <= 3:
                print("value changed, seed", seed, space.name, holomorphic, entry, before, after)
    print(f"{trials} Gateaux derivatives: {bad} values changed or outside N1; N1 rejected {rejected} of {trials} inputs; cases {rows}")
    return dict(value=bad, lazy=trials, rejected=rejected)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 400)


def test_gateaux():
    assert_clean(main(400))
