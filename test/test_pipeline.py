import os
import random
import sys

import numpy as np
from testutil import assert_clean

from uflx_spec_interpreter import actions, derivatives
from uflx_spec_interpreter.core import (
    Abs,
    Action,
    Conj,
    D,
    Div,
    Fn,
    Form,
    Im,
    Index,
    Indexed,
    ISum,
    Log,
    Num,
    Power,
    Prod,
    Re,
    Space,
    Sqrt,
    Sum,
    Tensor,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.static import arity

V, Vd = Space("V"), Space("V", dual=True)
u, v, w, z, h, q = (Fn(V, n) for n in "uvwzhq")
l = Fn(Vd, "l")
C = Prod(u, w)

VARIABLES = [u, w, C] if os.environ.get("COMPOUND") else [u, w]
POOL = [u, v, w, z, h, q, C]
BOUND = [v, q] + [Fn(V, f"y{k}") for k in range(6)]
i = Index("i")
c = Tensor((Num(1), Num(2)))


def scalar(rnd, depth, nd):
    if depth == 0 or rnd.random() < 0.3:
        pick = rnd.random()
        if pick < 0.15:
            return Num(rnd.choice([1, 2, -3, 0.5]))
        if pick < 0.25:
            return ISum(Prod(Indexed(c, (i,)), rnd.choice(POOL)), (i,))
        if pick < 0.32:
            return Action(l, 0, rnd.choice(POOL))
        return rnd.choice(POOL)
    a, b = scalar(rnd, depth - 1, nd), scalar(rnd, depth - 1, nd)
    near_one = Sum(Num(3), Prod(Num(0.2), a))
    ops = [
        lambda: Sum(a, b),
        lambda: Prod(a, b),
        lambda: Conj(a),
        lambda: Div(a, Sum(Num(4), Prod(Num(0.2), b))),
        lambda: Re(a),
        lambda: Im(a),
        lambda: Abs(near_one),
        lambda: Sqrt(near_one),
        lambda: Log(near_one),
        lambda: Power(near_one, rnd.choice([Num(2), Num(0.5), Prod(Num(0.3), b)])),
    ]
    if nd < 2:
        ops.append(lambda: derivative(rnd, Prod(rnd.choice(VARIABLES), a), nd + 1))
    return rnd.choice(ops)()


def derivative(rnd, f, nd):
    x = rnd.choice(VARIABLES)
    f = Prod(x, scalar(rnd, 2, nd))
    point = x if rnd.random() < 0.7 else rnd.choice([z, Sum(x, Num(0.1))])
    return D(f, x, point, rnd.choice([h, z, Num(1)]))


def form(rnd, n, depth):
    if depth == 0 or rnd.random() < 0.25:
        if n == 0:
            return scalar(rnd, 2, 0)
        names = rnd.sample(BOUND, n)
        body = scalar(rnd, 2, 0)
        for b in names:
            body = Prod(Conj(b) if rnd.random() < 0.3 else b, body)
        return Form(body, tuple(Conj(b) if rnd.random() < 0.3 else b for b in names))
    op = rnd.randrange(7)
    if op == 0:
        return Sum(form(rnd, n, depth - 1), form(rnd, n, depth - 1))
    if op == 1:
        k = rnd.randrange(n + 1)
        return Prod(form(rnd, k, depth - 1), form(rnd, n - k, depth - 1))
    if op == 2:
        return Conj(form(rnd, n, depth - 1))
    if op == 3:
        x = rnd.choice(VARIABLES)
        return D(Prod(x, form(rnd, n, depth - 1)), x, x, rnd.choice([h, z]))
    if op == 4:
        return ISum(Prod(Indexed(c, (i,)), form(rnd, n, depth - 1)), (i,))
    if op == 5:
        return Indexed(
            Tensor((form(rnd, n, depth - 1), form(rnd, n, depth - 1))), (rnd.randrange(2),)
        )
    return Action(form(rnd, n + 1, depth - 1), rnd.randrange(n + 1), scalar(rnd, 2, 0))


def main(trials):
    rng = np.random.default_rng(0)
    env = {f: 0.3 * complex(*rng.normal(size=2)) for f in POOL[:-1] + BOUND}
    cl = complex(*rng.normal(size=2))
    env[l] = lambda x: cl * np.conj(x)
    counts = dict(changed=0, not_n1=0, not_n2=0, derivatives=0, actions=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        n = rnd.randrange(3)
        e = form(rnd, n, 3)
        counts["derivatives"] += repr(e).count("D(f=")
        counts["actions"] += repr(e).count("Action(f=")
        out1 = actions.apply_actions(e)
        out2 = derivatives.apply_derivatives(out1)
        vals = tuple(0.3 * complex(*rng.normal(size=2)) for _ in range(arity(e)))
        before, after = evaluate(e, env)(vals), evaluate(out2, env)(vals)
        if not np.allclose(before, after, rtol=1e-3, atol=1e-4):
            counts["changed"] += 1
            if counts["changed"] <= 3:
                print("value changed, seed", seed, before, after)
        counts["not_n1"] += not actions.in_n2(out1)
        counts["not_n2"] += not (actions.in_n2(out2) and derivatives.in_n3(out2))
    print(
        f"{trials} expressions, {counts['actions']} actions, {counts['derivatives']} derivatives:",
        f"{counts['changed']} values changed, {counts['not_n1']} not in N2 after pass 1,",
        f"{counts['not_n2']} not in N3 after pass 2",
    )
    return dict(value=counts["changed"], grammar=counts["not_n1"] + counts["not_n2"])


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)


def test_pipeline():
    assert_clean(main(1000))
