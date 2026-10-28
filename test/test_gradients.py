import random
import sys

import numpy as np
from test_integrals import K
from testutil import assert_clean, contains

from uflx_spec_interpreter import oracle
from uflx_spec_interpreter.nodes import (
    CT,
    Abs,
    Conj,
    Coordinate,
    Div,
    Fn,
    Gradient,
    Index,
    Indexed,
    Integral,
    Inv,
    ISum,
    Log,
    Normal,
    Num,
    Power,
    Prod,
    PullBack,
    Re,
    Space,
    Sqrt,
    Sum,
    Tensor,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import gradients
from uflx_spec_interpreter.static import free, shape

X = Coordinate(K)
u = Fn(Space("V", domain=K), "u")
ENV = {u: lambda x: 1 + x[0] * x[0] + 0.5j * x[1]}
i, j = Index("i"), Index("j")


def near(a):
    return Sum(Num(3), Prod(Num(0.2), a))


def scalar(rnd, depth):
    if depth == 0 or rnd.random() < 0.3:
        return rnd.choice([Indexed(X, (0,)), Indexed(X, (1,)), u, Num(2)])
    a, b = scalar(rnd, depth - 1), scalar(rnd, depth - 1)
    return rnd.choice(
        [
            lambda: Sum(a, b),
            lambda: Prod(a, b),
            lambda: Div(a, near(b)),
            lambda: Conj(a),
            lambda: Re(a),
            lambda: Abs(near(a)),
            lambda: Sqrt(near(a)),
            lambda: Log(near(a)),
            lambda: Power(near(a), rnd.choice([Num(2), Prod(Num(0.3), b)])),
            lambda: Indexed(Gradient(a, K), (rnd.randrange(2),)),
            lambda: ISum(Prod(Indexed(Gradient(a, K), (i,)), Indexed(Gradient(b, K), (i,))), (i,)),
            lambda: Indexed(
                Inv(Tensor((Tensor((near(a), Prod(Num(0.1), b))), Tensor((Num(0.1), near(b)))))),
                (rnd.randrange(2), rnd.randrange(2)),
            ),
        ]
    )()


def target(rnd):
    # A gradient, a Hessian, or the gradient of a vector, of a random expression.
    a = scalar(rnd, 3)
    return rnd.choice(
        [
            lambda: Indexed(Gradient(a, K), (rnd.randrange(2),)),
            lambda: Indexed(Gradient(Gradient(a, K), K), (rnd.randrange(2), rnd.randrange(2))),
            lambda: Indexed(Gradient(Tensor((a, Prod(a, a))), K), (rnd.randrange(2), 0)),
            lambda: ISum(Indexed(Gradient(CT(Prod(a, Indexed(X, (j,))), (j,)), K), (i, i)), (i,)),
            lambda: Prod(a, Indexed(Gradient(X, K), (rnd.randrange(2), rnd.randrange(2)))),
        ]
    )()


def needs_pass(n):
    return isinstance(n, Gradient) and not isinstance(n.b, (Fn, Normal, PullBack, Gradient))


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
        e = Integral(target(rnd), K)
        out = gradients.apply_gradients(e)
        if contains(e, needs_pass):
            bad["lazy"] += 1
            bad["rejected"] += not gradients.in_n6(e)
        if (shape(e), free(e)) != (shape(out), free(out)):
            bad["static"] += 1
        bad["grammar"] += not gradients.in_n6(out)
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

    pi = Gradient(Coordinate(M), M)
    assert gradients.apply_gradients(pi) == pi
    assert gradients.in_n6(pi)
