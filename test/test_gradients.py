import random
import sys

import numpy as np
from test_integrals import K
from testutil import assert_clean, check_types, contains, keeps_static_type

import uflx_oracle as oracle
from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    absolute,
    add,
    component_tensor,
    conjugate,
    coordinate,
    divide,
    expand_arithmetic,
    function,
    gradient,
    index,
    index_sum,
    indexed,
    integral,
    inverse,
    logarithm,
    multiply,
    num,
    power,
    real_part,
    square_root,
    tensor,
)
from uflx_spec_interpreter._src.nodes import Function, Gradient, Normal, PullBack
from uflx_spec_interpreter._src.passes import gradients
from uflx_spec_interpreter._src.passes.contracts import _in_core_normal_form
from uflx_toy.spaces import space

X = coordinate(K)
u = function(space("V", domain=K), "u")
ENV = {u: lambda x: 1 + x[0] * x[0] + 0.5j * x[1]}
i, j = index("i"), index("j")


def near(a):
    return add(num(3), multiply(num(0.2), a))


def scalar(rnd, depth):
    if depth == 0 or rnd.random() < 0.3:
        return rnd.choice([indexed(X, (0,)), indexed(X, (1,)), u, num(2)])
    a, b = scalar(rnd, depth - 1), scalar(rnd, depth - 1)
    return rnd.choice(
        [
            lambda: add(a, b),
            lambda: multiply(a, b),
            lambda: divide(a, near(b)),
            lambda: conjugate(a),
            lambda: real_part(a),
            lambda: absolute(near(a)),
            lambda: square_root(near(a)),
            lambda: logarithm(near(a)),
            lambda: power(near(a), rnd.choice([num(2), multiply(num(0.3), b)])),
            lambda: indexed(gradient(a, K), (rnd.randrange(2),)),
            lambda: index_sum(
                multiply(indexed(gradient(a, K), (i,)), indexed(gradient(b, K), (i,))), (i,)
            ),
            lambda: indexed(
                inverse(
                    tensor((tensor((near(a), multiply(num(0.1), b))), tensor((num(0.1), near(b)))))
                ),
                (rnd.randrange(2), rnd.randrange(2)),
            ),
        ]
    )()


def target(rnd):
    # A gradient, a Hessian, or the gradient of a vector, of a random expression.
    a = scalar(rnd, 3)
    return rnd.choice(
        [
            lambda: indexed(gradient(a, K), (rnd.randrange(2),)),
            lambda: indexed(gradient(gradient(a, K), K), (rnd.randrange(2), rnd.randrange(2))),
            lambda: indexed(gradient(tensor((a, multiply(a, a))), K), (rnd.randrange(2), 0)),
            lambda: index_sum(
                indexed(gradient(component_tensor(multiply(a, indexed(X, (j,))), (j,)), K), (i, i)),
                (i,),
            ),
            lambda: multiply(a, indexed(gradient(X, K), (rnd.randrange(2), rnd.randrange(2)))),
        ]
    )()


def needs_pass(n):
    return isinstance(n, Gradient) and not isinstance(
        n.expression, (Function, Normal, PullBack, Gradient)
    )


def main(trials):
    # Third derivatives by nested finite differences need a larger step than the default.
    step, oracle.H = oracle.H, 1e-3
    oracle.VALUES.clear()
    try:
        return run(trials)
    finally:
        oracle.H = step
        oracle.VALUES.clear()


def run(trials):
    bad = dict(value=0, static=0, grammar=0, lazy=0, rejected=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        e = integral(target(rnd), K)
        out = gradients.apply_gradients(e)
        out = expand_arithmetic(out)
        if contains(e, needs_pass):
            bad["lazy"] += 1
            bad["rejected"] += not _in_core_normal_form(e)
        check_types(bad, e, out)
        if not keeps_static_type(e, out):
            bad["static"] += 1
        bad["grammar"] += not _in_core_normal_form(out)
        before, after = evaluate(e, ENV)(()), evaluate(out, ENV)(())
        if not np.allclose(before, after, rtol=1e-4, atol=1e-5):
            bad["value"] += 1
            if bad["value"] <= 3:
                print("value changed, seed", seed, before, after)
    print(
        f"{trials} gradients on a reference cell: {bad['value']} values changed, "
        f"{bad['static']} static types changed, {bad['grammar']} outside N6; "
        f"N6 rejected {bad['rejected']} of {bad['lazy']} inputs with a gradient to evaluate"
    )
    return bad


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)


def test_gradients():
    assert_clean(main(200))


def test_projection_remains():
    # On a surface the gradient of the coordinates is the projection, which needs a cell.
    from test_surface import M

    pi = gradient(coordinate(M), M)
    assert gradients.apply_gradients(pi) == pi
    assert _in_core_normal_form(pi)
