import random
import sys

import numpy as np
from test_facets import table
from test_integrals import D, u
from test_surface import M
from testutil import assert_clean, contains

from uflx_spec_interpreter.domains import CoordDomain, Decomposed, DomainMap
from uflx_spec_interpreter.nodes import (
    CT,
    Coordinate,
    Fn,
    Gradient,
    Index,
    Indexed,
    Integral,
    Num,
    Prod,
    PullBack,
    Space,
    Sum,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import gradients, integrals, pullbacks
from uflx_spec_interpreter.static import free, shape

i, k = Index("i"), Index("k")
POINT = CoordDomain("pt", 0)


def points(coordinates, Q):
    # A family of points in Q, with the free index i.
    return Decomposed((DomainMap(POINT, CT(Indexed(table(coordinates), (i, k)), (k,)), Q),))


def on_surface(cell, X):
    # A point of the folded surface M, from a reference point of one of its two triangles.
    verts = [[(0, 0, 0), (1, 0, 0), (1, 1, 0.5)], [(0, 0, 0), (1, 1, 0.5), (0, 1, 1)]][cell]
    v0, v1, v2 = (np.array(v, dtype=float) for v in verts)
    return tuple(v0 + (v1 - v0) * X[0] + (v2 - v0) * X[1])


SQUARE_POINTS = points([(0.2, 0.3), (0.7, 0.4), (0.35, 0.8), (0.9, 0.95)], D)
SURFACE_POINTS = points([on_surface(0, (0.3, 0.2)), on_surface(1, (0.25, 0.5))], M)
w = Fn(Space("W", domain=M), "w")
ENV = {u: lambda x: 1 + x[0] + 2 * x[0] * x[1], w: lambda x: 1 + x[0] * x[1] + x[2] * x[2]}


def integrand(rnd, Q, f):
    X = Coordinate(Q)
    n = 2 if Q == D else 3
    r, s = rnd.randrange(n), rnd.randrange(n)
    grad = lambda g: Gradient(g, Q)
    choices = [
        f,
        Indexed(X, (r,)),
        Indexed(grad(f), (r,)),
        Indexed(grad(Prod(f, f)), (r,)),
        Indexed(grad(Prod(f, Indexed(X, (0,)))), (r,)),
        Indexed(grad(X), (r, s)),
    ]
    if Q == D:
        # Hessians of a terminal and of a product, both evaluated on the mesh.
        choices += [Indexed(grad(grad(f)), (r, s)), Indexed(grad(grad(Prod(f, f))), (r, s))]
    a, b = rnd.choice(choices), rnd.choice(choices)
    return rnd.choice([a, Sum(a, b), Prod(a, b), Prod(Num(2), a)])


def stays_for_the_mesh_library(n):
    # A pull-back by the points of a function or of its gradient remains for the mesh library.
    return isinstance(n, PullBack) and isinstance(n.b, (Fn, Gradient))


def needs_pass(n):
    return isinstance(n, Gradient) and not isinstance(n.b, (Fn, Coordinate, Gradient, PullBack))


def main(trials):
    bad = dict(value=0, static=0, grammar=0, lazy=0, rejected=0, unlocated=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        Q, family, f = rnd.choice([(D, SQUARE_POINTS, u), (M, SURFACE_POINTS, w)])
        e = Integral(integrand(rnd, Q, f), family)
        out4 = pullbacks.expand_pull_backs(integrals.expand_integrals(e))
        out = gradients.apply_gradients(out4)
        if contains(
            out4,
            lambda n: (
                isinstance(n, Gradient)
                and not isinstance(n.b, (Fn, Coordinate, Gradient, PullBack))
            ),
        ):
            bad["lazy"] += 1
            bad["rejected"] += not gradients.in_n6(out4)
        if (shape(e), free(e)) != (shape(out), free(out)):
            bad["static"] += 1
        bad["grammar"] += not (pullbacks.in_n5(out) and gradients.in_n6(out))
        bad["unlocated"] += contains(e, lambda n: isinstance(n, Fn)) and not contains(
            out, stays_for_the_mesh_library
        )
        before, after = evaluate(e, ENV)(()), evaluate(out, ENV)(())
        if not np.allclose(before, after, rtol=1e-4, atol=1e-5):
            bad["value"] += 1
            if bad["value"] <= 3:
                print("value changed, seed", seed, before, after)
    print(
        f"{trials} integrals over points: {bad['value']} values changed, "
        f"{bad['static']} static types changed, {bad['grammar']} outside N5 or N6, "
        f"{bad['unlocated']} with no pull-back left for the mesh library; "
        f"N6 rejected {bad['rejected']} of {bad['lazy']} inputs with a gradient to evaluate"
    )
    return bad


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)


def test_points():
    counts = main(60)
    assert counts["unlocated"] == 0
    assert_clean(counts)
