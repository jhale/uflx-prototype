import random
import sys

import numpy as np

from uflx._src.nodes import Conormal, Function, Gradient, PullBack
from uflx._src.passes import gradients
from uflx._src.passes.contracts import in_core_normal_form
from uflx.arithmetic import (
    absolute,
    add,
    conjugate,
    divide,
    logarithm,
    multiply,
    num,
    power,
    real_part,
    square_root,
)
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import function
from uflx.passes import expand_arithmetic
from uflx.tensors import component_tensor, index, index_sum, indexed, inverse, tensor

from ..testutil import assert_clean, check_types, contains, keeps_static_type
from ..uflx.spaces import space
from ..uflx_basix.integrals import K
from .quadrature import integrate_at

X = coordinate(K)
u = function(space("V", domain=K), "u")
ENV = {u: num(1) + X[0] * X[0] + num(0.5j) * X[1]}
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
        n.expression, (Function, Conormal, PullBack, Gradient)
    )


def main(trials):
    bad = dict(value=0, static=0, grammar=0, lazy=0, rejected=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        e = integral(target(rnd), K)
        out = gradients.apply_gradients(e)
        out = expand_arithmetic(out)
        if contains(e, needs_pass):
            bad["lazy"] += 1
            bad["rejected"] += not in_core_normal_form(e)
        check_types(bad, e, out)
        if not keeps_static_type(e, out):
            bad["static"] += 1
        bad["grammar"] += not in_core_normal_form(out)
        before, after = integrate_at(e, ENV, degree=12), integrate_at(out, ENV, degree=12)
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
