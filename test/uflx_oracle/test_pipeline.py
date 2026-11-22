import os
import random
import sys

import numpy as np

from uflx._src.passes import applications, derivatives
from uflx._src.passes.applications import applications_expanded
from uflx._src.passes.derivatives import _derivatives_complete
from uflx.arithmetic import (
    absolute,
    add,
    conjugate,
    divide,
    imaginary_part,
    logarithm,
    multiply,
    num,
    power,
    real_part,
    square_root,
)
from uflx.calculus import directional_derivative
from uflx.forms import apply, dual_space, form, function
from uflx.tensors import index, index_sum, indexed, tensor
from uflx_oracle.interpreter import Interpreter

from ..testutil import assert_clean, check_types, form_with_fixed_argument
from ..uflx.spaces import space

V, Vd = space("V"), dual_space(space("V"))
u, v, w, z, h, q = (function(V, n) for n in "uvwzhq")
l = function(Vd, "l")
C = multiply(u, w)

VARIABLES = [u, w, C] if os.environ.get("COMPOUND") else [u, w]
POOL = [u, v, w, z, h, q, C]
BOUND = [v, q] + [function(V, f"y{k}") for k in range(6)]
i = index("i")
c = tensor((num(1), num(2)))


def scalar(rnd, depth, nd):
    if depth == 0 or rnd.random() < 0.3:
        pick = rnd.random()
        if pick < 0.15:
            return num(rnd.choice([1, 2, -3, 0.5]))
        if pick < 0.25:
            return index_sum(multiply(indexed(c, (i,)), rnd.choice(POOL)), (i,))
        if pick < 0.32:
            return apply(l, (rnd.choice(POOL),))
        return rnd.choice(POOL)
    a, b = scalar(rnd, depth - 1, nd), scalar(rnd, depth - 1, nd)
    near_one = add(num(3), multiply(num(0.2), a))
    ops = [
        lambda: add(a, b),
        lambda: multiply(a, b),
        lambda: conjugate(a),
        lambda: divide(a, add(num(4), multiply(num(0.2), b))),
        lambda: real_part(a),
        lambda: imaginary_part(a),
        lambda: absolute(near_one),
        lambda: square_root(near_one),
        lambda: logarithm(near_one),
        lambda: power(near_one, rnd.choice([num(2), num(0.5), multiply(num(0.3), b)])),
    ]
    if nd < 2:
        ops.append(lambda: derivative(rnd, multiply(rnd.choice(VARIABLES), a), nd + 1))
    return rnd.choice(ops)()


def derivative(rnd, f, nd):
    x = rnd.choice(VARIABLES)
    f = multiply(x, scalar(rnd, 2, nd))
    point = x if rnd.random() < 0.7 else rnd.choice([z, add(x, num(0.1))])
    return directional_derivative(f, x, point, rnd.choice([h, z, num(1)]))


def random_form(rnd, kinds, depth):
    # One kind for each argument, True for an entry conjugate(v), which is antilinear.
    n = len(kinds)
    if depth == 0 or rnd.random() < 0.25:
        if n == 0:
            return scalar(rnd, 2, 0)
        names = rnd.sample(BOUND, n)
        body = scalar(rnd, 2, 0)
        for b in names:
            body = multiply(conjugate(b) if rnd.random() < 0.3 else b, body)
        entries = tuple(conjugate(b) if k else b for b, k in zip(names, kinds, strict=True))
        return form(body, entries)
    op = rnd.randrange(7)
    if op == 0:
        return add(random_form(rnd, kinds, depth - 1), random_form(rnd, kinds, depth - 1))
    if op == 1:
        k = rnd.randrange(n + 1)
        first, second = kinds[:k], kinds[k:]
        return multiply(random_form(rnd, first, depth - 1), random_form(rnd, second, depth - 1))
    if op == 2:
        return conjugate(random_form(rnd, tuple(not k for k in kinds), depth - 1))
    if op == 3:
        x = rnd.choice(VARIABLES)
        return directional_derivative(
            multiply(x, random_form(rnd, kinds, depth - 1)), x, x, rnd.choice([h, z])
        )
    if op == 4:
        return index_sum(multiply(indexed(c, (i,)), random_form(rnd, kinds, depth - 1)), (i,))
    if op == 5:
        return indexed(
            tensor((random_form(rnd, kinds, depth - 1), random_form(rnd, kinds, depth - 1))),
            (rnd.randrange(2),),
        )
    p = rnd.randrange(n + 1)
    inner = (*kinds[:p], rnd.random() < 0.3, *kinds[p:])
    return form_with_fixed_argument(random_form(rnd, inner, depth - 1), p, scalar(rnd, 2, 0))


def main(trials):
    rng = np.random.default_rng(0)
    env = {f: 0.3 * complex(*rng.normal(size=2)) for f in POOL[:-1] + BOUND}
    cl = complex(*rng.normal(size=2))
    env[l] = lambda x: cl * x
    counts = dict(changed=0, not_n1=0, not_n2=0, derivatives=0, actions=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        kinds = tuple(rnd.random() < 0.3 for _ in range(rnd.randrange(3)))
        e = random_form(rnd, kinds, 3)
        counts["derivatives"] += repr(e).count("DirectionalDerivative(")
        counts["actions"] += repr(e).count("Apply(")
        out1 = applications.expand_applications(e)
        out2 = derivatives.apply_derivatives(out1)
        check_types(counts, e, out1, out2)
        vals = tuple(0.3 * complex(*rng.normal(size=2)) for _ in range(len(e.argument_spaces)))
        oracle = Interpreter(env)
        before, after = oracle.evaluate(e, arguments=vals), oracle.evaluate(out2, arguments=vals)
        if not np.allclose(before, after, rtol=1e-3, atol=1e-4):
            counts["changed"] += 1
            if counts["changed"] <= 3:
                print("value changed, seed", seed, before, after)
        counts["not_n1"] += not applications_expanded(out1)
        counts["not_n2"] += not (applications_expanded(out2) and _derivatives_complete(out2))
    print(
        f"{trials} expressions, {counts['actions']} actions, {counts['derivatives']} derivatives:",
        f"{counts['changed']} values changed, {counts['not_n1']} not in N2 after pass 1,",
        f"{counts['not_n2']} not in N3 after pass 2",
    )
    return dict(
        value=counts["changed"],
        grammar=counts["not_n1"] + counts["not_n2"],
        typed=counts.get("typed", 0),
    )


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1000)


def test_pipeline():
    assert_clean(main(1000))
