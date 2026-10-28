import random
import sys

import numpy as np
from testutil import assert_clean, contains, pairing

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    action,
    add,
    conjugate,
    directional_derivative,
    divide,
    form,
    function,
    index,
    index_sum,
    indexed,
    multiply,
    num,
    tensor,
)
from uflx_spec_interpreter._src.nodes import Action
from uflx_spec_interpreter._src.passes import actions
from uflx_spec_interpreter._src.static import arity
from uflx_toy.spaces import space

V = space("V")
u, v, w, z, h, q = (function(V, n) for n in "uvwzhq")
U = function(space("W", value_shape=(2,)), "U")
POOL = [u, v, w, z, h, q]
BOUND = [v, q, function(V, "y1"), function(V, "y2"), function(V, "y3")]
i = index("i")
c = tensor((num(1), num(2)))


def scalar(rnd, depth):
    if depth == 0 or rnd.random() < 0.3:
        pick = rnd.random()
        if pick < 0.15:
            return num(rnd.choice([1, 2, -3, 0.5]))
        if pick < 0.25:
            return indexed(U, (rnd.randrange(2),))
        if pick < 0.35:
            return index_sum(multiply(indexed(c, (i,)), rnd.choice(POOL)), (i,))
        return rnd.choice(POOL)
    op = rnd.randrange(5)
    a, b = scalar(rnd, depth - 1), scalar(rnd, depth - 1)
    if op == 0:
        return add(a, b)
    if op == 1:
        return multiply(a, b)
    if op == 2:
        return conjugate(a)
    if op == 3:
        return divide(a, num(rnd.choice([2, -4])))
    return directional_derivative(multiply(u, a), u, h, u)


def random_form(rnd, n, depth):
    if depth == 0 or rnd.random() < 0.25:
        if n == 0:
            return scalar(rnd, 2)
        names = rnd.sample(BOUND, n)
        body = scalar(rnd, 2)
        for b in names:
            body = multiply(conjugate(b) if rnd.random() < 0.3 else b, body)
        entries = tuple(conjugate(b) if rnd.random() < 0.3 else b for b in names)
        return form(body, entries)
    op = rnd.randrange(8)
    if op == 0:
        return add(random_form(rnd, n, depth - 1), random_form(rnd, n, depth - 1))
    if op == 1:
        k = rnd.randrange(n + 1)
        return multiply(random_form(rnd, k, depth - 1), random_form(rnd, n - k, depth - 1))
    if op == 2:
        return divide(random_form(rnd, n, depth - 1), num(3))
    if op == 3:
        return conjugate(random_form(rnd, n, depth - 1))
    if op == 4:
        return directional_derivative(random_form(rnd, n, depth - 1), u, h, u)
    if op == 5:
        return indexed(
            tensor((random_form(rnd, n, depth - 1), random_form(rnd, n, depth - 1))),
            (rnd.randrange(2),),
        )
    if op == 6:
        return index_sum(multiply(indexed(c, (i,)), random_form(rnd, n, depth - 1)), (i,))
    return action(random_form(rnd, n + 1, depth - 1), rnd.randrange(n + 1), scalar(rnd, 2))


def main(trials):
    rng = np.random.default_rng(0)
    env = {f: complex(*rng.normal(size=2)) for f in POOL + BOUND}
    env[U] = rng.normal(size=2) + 1j * rng.normal(size=2)
    failures, not_n1, n_actions, lazy, rejected = 0, 0, 0, 0, 0
    for seed in range(trials):
        rnd = random.Random(seed)
        n = rnd.randrange(3)
        e = random_form(rnd, n, 4)
        n_actions += str(e).count("Action(")
        out = actions.apply_actions(e)
        vals = tuple(complex(*rng.normal(size=2)) for _ in range(arity(e)))
        before, after = evaluate(e, env)(vals), evaluate(out, env)(vals)
        if not np.allclose(before, after, rtol=1e-4, atol=1e-4):
            failures += 1
            if failures == 1:
                print("first failure, seed", seed, before, after)
        if contains(
            e,
            lambda n: isinstance(n, Action) and not pairing(n),
        ):
            lazy += 1
            rejected += not actions.in_n2(e)
        if not actions.in_n2(out):
            not_n1 += 1
    print(
        f"{trials} expressions with {n_actions} actions: {failures} values changed, "
        f"{not_n1} not in N2; "
        f"N2 rejected {rejected} of {lazy} inputs with an action to apply"
    )
    return dict(value=failures, grammar=not_n1, lazy=lazy, rejected=rejected)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2000)


def test_random():
    assert_clean(main(2000))
