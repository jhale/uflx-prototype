import random
import sys

import numpy as np

from uflx_spec_interpreter import actions
from uflx_spec_interpreter.core import CT, D, Action, Conj, Div, Fn, Form, Index, Indexed, ISum, Num, Prod, Space, Sum, Tensor
from uflx_spec_interpreter.static import arity
from uflx_spec_interpreter.oracle import evaluate
from testutil import contains
from testutil import assert_clean

V = Space("V")
u, v, w, z, h, q = (Fn(V, n) for n in "uvwzhq")
U = Fn(Space("W", shape=(2,)), "U")
POOL = [u, v, w, z, h, q]
BOUND = [v, q, Fn(V, "y1"), Fn(V, "y2"), Fn(V, "y3")]
i = Index("i")
c = Tensor((Num(1), Num(2)))


def scalar(rnd, depth):
    if depth == 0 or rnd.random() < 0.3:
        pick = rnd.random()
        if pick < 0.15:
            return Num(rnd.choice([1, 2, -3, 0.5]))
        if pick < 0.25:
            return Indexed(U, (rnd.randrange(2),))
        if pick < 0.35:
            return ISum(Prod(Indexed(c, (i,)), rnd.choice(POOL)), (i,))
        return rnd.choice(POOL)
    op = rnd.randrange(5)
    a, b = scalar(rnd, depth - 1), scalar(rnd, depth - 1)
    if op == 0:
        return Sum(a, b)
    if op == 1:
        return Prod(a, b)
    if op == 2:
        return Conj(a)
    if op == 3:
        return Div(a, Num(rnd.choice([2, -4])))
    return D(Prod(u, a), u, u, h)


def form(rnd, n, depth):
    if depth == 0 or rnd.random() < 0.25:
        if n == 0:
            return scalar(rnd, 2)
        names = rnd.sample(BOUND, n)
        body = scalar(rnd, 2)
        for b in names:
            body = Prod(Conj(b) if rnd.random() < 0.3 else b, body)
        entries = tuple(Conj(b) if rnd.random() < 0.3 else b for b in names)
        return Form(body, entries)
    op = rnd.randrange(8)
    if op == 0:
        return Sum(form(rnd, n, depth - 1), form(rnd, n, depth - 1))
    if op == 1:
        k = rnd.randrange(n + 1)
        return Prod(form(rnd, k, depth - 1), form(rnd, n - k, depth - 1))
    if op == 2:
        return Div(form(rnd, n, depth - 1), Num(3))
    if op == 3:
        return Conj(form(rnd, n, depth - 1))
    if op == 4:
        return D(form(rnd, n, depth - 1), u, u, h)
    if op == 5:
        return Indexed(Tensor((form(rnd, n, depth - 1), form(rnd, n, depth - 1))), (rnd.randrange(2),))
    if op == 6:
        return ISum(Prod(Indexed(c, (i,)), form(rnd, n, depth - 1)), (i,))
    return Action(form(rnd, n + 1, depth - 1), rnd.randrange(n + 1), scalar(rnd, 2))


def main(trials):
    rng = np.random.default_rng(0)
    env = {f: complex(*rng.normal(size=2)) for f in POOL + BOUND}
    env[U] = rng.normal(size=2) + 1j * rng.normal(size=2)
    failures, not_n1, n_actions, lazy, rejected = 0, 0, 0, 0, 0
    for seed in range(trials):
        rnd = random.Random(seed)
        n = rnd.randrange(3)
        e = form(rnd, n, 4)
        n_actions += str(e).count("Action(")
        out = actions.apply_actions(e)
        vals = tuple(complex(*rng.normal(size=2)) for _ in range(arity(e)))
        before, after = evaluate(e, env)(vals), evaluate(out, env)(vals)
        if not np.allclose(before, after, rtol=1e-4, atol=1e-4):
            failures += 1
            if failures == 1:
                print("first failure, seed", seed, before, after)
        if contains(e, lambda n: isinstance(n, Action) and not (isinstance(n.f, Fn) and n.f.space.dual)):
            lazy += 1
            rejected += not actions.in_n2(e)
        if not actions.in_n2(out):
            not_n1 += 1
    print(f"{trials} expressions with {n_actions} actions: {failures} values changed, {not_n1} not in N2; N2 rejected {rejected} of {lazy} inputs with an action to apply")
    return dict(value=failures, grammar=not_n1, lazy=lazy, rejected=rejected)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)


def test_random():
    assert_clean(main(2000))
