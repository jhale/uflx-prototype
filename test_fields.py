import random
import sys

import numpy as np

import derivatives
import integrals
import passes
from core import CT, D, Action, Conj, Coordinate, Div, Fn, Form, Gradient, Index, Indexed, Integral, ISum, Num, Prod, PullBack, Space, Sum, cell_domain
from oracle import evaluate
from static import arity

from test_integrals import D as MESH
from test_integrals import K

VD = Space("VD", domain=MESH)
u, w, h = (Fn(VD, n) for n in "uwh")
BOUND = [Fn(VD, n) for n in ("v", "y0", "y1", "y2")]
i, c = Index("i"), Index("c")
X = Coordinate(MESH)


def field(rnd, names, depth):
    # A scalar field on the mesh, built from the names, u, w and the coordinates.
    if depth == 0 or rnd.random() < 0.35:
        base = rnd.choice(names + [u, w, Indexed(X, (rnd.randrange(2),)), Num(2)])
        if rnd.random() < 0.3 and isinstance(base, Fn):
            return Indexed(Gradient(base, MESH), (rnd.randrange(2),))
        return base
    a, b = field(rnd, names, depth - 1), field(rnd, names, depth - 1)
    return rnd.choice([
        Sum(a, b),
        Prod(a, b),
        Conj(a),
        Div(a, Sum(Num(3), Prod(Num(0.2), b))),
        ISum(Prod(Indexed(Gradient(a, MESH), (i,)), Indexed(Gradient(b, MESH), (i,))), (i,)),
    ])


def integral_of(rnd, body):
    # The integral over the mesh, or the same written by hand with a pull-back by the cells.
    if rnd.random() < 0.7:
        return Integral(body, MESH)
    cells = cell_domain(MESH, 0)
    return ISum(Integral(Prod(PullBack(body, cells), integrals.density(cells)), K), (c,))


def form(rnd, n, depth):
    names = rnd.sample(BOUND, n)
    body = field(rnd, [], 2)
    for b in names:
        body = Prod(b, body)
    e = Form(integral_of(rnd, body), tuple(names)) if names else integral_of(rnd, body)
    if depth > 0 and rnd.random() < 0.5:
        e = D(e, u, u, rnd.choice([h, w]))
    if depth > 0 and rnd.random() < 0.6:
        bigger = form(rnd, n + 1, depth - 1)
        e = Sum(e, Action(bigger, rnd.randrange(n + 1), field(rnd, [], 1)))
    return e


def main(trials):
    env = {u: lambda x: 1 + x[0] * x[1], w: lambda x: x[0] - 2 * x[1], h: lambda x: 0.5 + x[1]}
    args = [lambda x: 1 + x[0], lambda x: x[1] * x[1]]
    bad = dict(value=0, grammar=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        n = rnd.randrange(2)
        e = form(rnd, n, 2)
        out = integrals.expand_integrals(derivatives.apply_derivatives(passes.apply_actions(e)))
        vals = tuple(args[:arity(e)])
        before, after = evaluate(e, env)(vals), evaluate(out, env)(vals)
        if not np.allclose(before, after, rtol=1e-4, atol=1e-6):
            bad["value"] += 1
            if bad["value"] <= 3:
                print("value changed, seed", seed, before, after)
        bad["grammar"] += not (passes.in_n2(out) and derivatives.in_n3(out) and integrals.in_n4(out))
    print(f"{trials} forms on a mesh: {bad['value']} values changed, {bad['grammar']} outside N4")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)
