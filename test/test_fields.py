import random
import sys

import numpy as np
from test_integrals import D as MESH
from test_integrals import K
from testutil import assert_clean, check_types

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    action,
    add,
    cell_domain,
    conjugate,
    coordinate,
    density,
    directional_derivative,
    divide,
    form,
    function,
    gradient,
    index,
    index_sum,
    indexed,
    integral,
    multiply,
    num,
    pull_back,
)
from uflx_spec_interpreter._src.nodes import Function
from uflx_spec_interpreter._src.passes import actions, derivatives, integrals
from uflx_spec_interpreter._src.static import arity
from uflx_toy.spaces import space

VD = space("VD", domain=MESH)
u, w, h = (function(VD, n) for n in "uwh")
BOUND = [function(VD, n) for n in ("v", "y0", "y1", "y2")]
i, c = index("i"), index("c")
X = coordinate(MESH)


def field(rnd, names, depth):
    # A scalar field on the mesh, built from the names, u, w and the coordinates.
    if depth == 0 or rnd.random() < 0.35:
        base = rnd.choice([*names, u, w, indexed(X, (rnd.randrange(2),)), num(2)])
        if rnd.random() < 0.3 and isinstance(base, Function):
            return indexed(gradient(base, MESH), (rnd.randrange(2),))
        return base
    a, b = field(rnd, names, depth - 1), field(rnd, names, depth - 1)
    return rnd.choice(
        [
            add(a, b),
            multiply(a, b),
            conjugate(a),
            divide(a, add(num(3), multiply(num(0.2), b))),
            index_sum(
                multiply(indexed(gradient(a, MESH), (i,)), indexed(gradient(b, MESH), (i,))), (i,)
            ),
        ]
    )


def integral_of(rnd, body):
    # The integral over the mesh, or the same written by hand with a pull-back by the cells.
    if rnd.random() < 0.7:
        return integral(body, MESH)
    cells = cell_domain(MESH, 0)
    return index_sum(integral(multiply(pull_back(body, cells), density(cells)), K), (c,))


def random_form(rnd, n, depth):
    names = rnd.sample(BOUND, n)
    body = field(rnd, [], 2)
    for b in names:
        body = multiply(b, body)
    e = form(integral_of(rnd, body), tuple(names)) if names else integral_of(rnd, body)
    if depth > 0 and rnd.random() < 0.5:
        e = directional_derivative(e, u, rnd.choice([h, w]), u)
    if depth > 0 and rnd.random() < 0.6:
        bigger = random_form(rnd, n + 1, depth - 1)
        e = add(e, action(bigger, rnd.randrange(n + 1), field(rnd, [], 1)))
    return e


def main(trials):
    env = {u: lambda x: 1 + x[0] * x[1], w: lambda x: x[0] - 2 * x[1], h: lambda x: 0.5 + x[1]}
    args = [lambda x: 1 + x[0], lambda x: x[1] * x[1]]
    bad = dict(value=0, grammar=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        n = rnd.randrange(2)
        e = random_form(rnd, n, 2)
        out = integrals.expand_integrals(derivatives.apply_derivatives(actions.apply_actions(e)))
        check_types(bad, e, out)
        vals = tuple(args[: arity(e)])
        before, after = evaluate(e, env)(vals), evaluate(out, env)(vals)
        if not np.allclose(before, after, rtol=1e-4, atol=1e-6):
            bad["value"] += 1
            if bad["value"] <= 3:
                print("value changed, seed", seed, before, after)
        bad["grammar"] += not (
            actions.in_n2(out) and derivatives.in_n3(out) and integrals.in_n4(out)
        )
    print(f"{trials} forms on a mesh: {bad['value']} values changed, {bad['grammar']} outside N4")
    return bad


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)


def test_fields():
    assert_clean(main(60))
