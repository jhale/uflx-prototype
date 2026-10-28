import itertools
import random
import sys

import numpy as np
from testutil import assert_clean

from uflx_spec_interpreter.domains import CoordDomain, DecomposedDomain, DomainMap
from uflx_spec_interpreter.nodes import (
    ComponentTensor,
    Coordinate,
    Function,
    Gradient,
    Index,
    Indexed,
    Integral,
    Num,
    Product,
    Space,
    Sum,
    Tensor,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import gradients, integrals, pullbacks
from uflx_spec_interpreter.static import free, shape

R2 = CoordDomain("R2", 2)
K = CoordDomain("K", 2)
X = Coordinate(K)
k = Index("k")
c = Index("c")
num = lambda v: Num(v)


def affine(vertices, at):
    # The affine map of the unit triangle onto a triangle, as an expression of the coordinate at.
    v0, v1, v2 = (Tensor(tuple(num(x) for x in v)) for v in vertices)
    term = lambda v: Indexed(v, (k,))
    body = Sum(
        term(v0),
        Sum(
            Product(Sum(term(v1), Product(num(-1), term(v0))), Indexed(at, (0,))),
            Product(Sum(term(v2), Product(num(-1), term(v0))), Indexed(at, (1,))),
        ),
    )
    return ComponentTensor(body, (k,))


T0, T1 = [(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]

# The square as one family of two triangles, with the free index c.
p = Tensor(tuple(Tensor(tuple(Tensor((num(a), num(b))) for a, b in tri)) for tri in (T0, T1)))
fam = lambda i: Indexed(p, (c, i, k))
psi_c = ComponentTensor(
    Sum(
        fam(0),
        Sum(
            Product(Sum(fam(1), Product(num(-1), fam(0))), Indexed(X, (0,))),
            Product(Sum(fam(2), Product(num(-1), fam(0))), Indexed(X, (1,))),
        ),
    ),
    (k,),
)
D = DecomposedDomain((DomainMap(K, psi_c, R2),))

# A family of three points in the square.
Dp = DecomposedDomain(
    (
        DomainMap(
            CoordDomain("pt", 0),
            ComponentTensor(
                Indexed(
                    Tensor(
                        tuple(
                            Tensor((num(a), num(b)))
                            for a, b in [(0.2, 0.3), (0.5, 0.5), (0.9, 0.1)]
                        )
                    ),
                    (c, k),
                ),
                (k,),
            ),
            R2,
        ),
    )
)

# The square as two pieces with no free index.
D2 = DecomposedDomain((DomainMap(K, affine(T0, X), R2), DomainMap(K, affine(T1, X), R2)))

# A macro triangle: T0 as the image of the unit triangle split into two sub-triangles.
Kdec = DecomposedDomain(
    (
        DomainMap(K, affine([(0, 0), (1, 0), (0.5, 0.5)], X), K),
        DomainMap(K, affine([(0, 0), (0.5, 0.5), (0, 1)], X), K),
    )
)
Dm = DecomposedDomain((DomainMap(Kdec, affine(T0, Coordinate(Kdec)), R2),))

V = Space("V", domain=D)
u = Function(V, "u")
r = Function(Space("R"), "r")
d = Tensor((num(3), num(5)))


def scalar(rnd, dom, kind, depth):
    if depth == 0 or rnd.random() < 0.35:
        leaves = {
            "domain": [Indexed(Coordinate(dom), (0,)), Indexed(Coordinate(dom), (1,))]
            + ([u, Indexed(Gradient(u, D), (rnd.randrange(2),))] if dom == D else []),
            "ambient": [Indexed(Coordinate(R2), (0,)), Indexed(Coordinate(R2), (1,))]
            + ([Indexed(d, (c,))] if dom == D else []),
            "none": [num(rnd.choice([1, 2, -0.5])), r],
        }[kind]
        return rnd.choice([*leaves, num(2)])
    a, b = scalar(rnd, dom, kind, depth - 1), scalar(rnd, dom, kind, depth - 1)
    return rnd.choice([Sum(a, b), Product(a, b)])


def integrand(rnd, dom, kind):
    pick = rnd.randrange(4)
    if pick == 0:
        return Tensor((scalar(rnd, dom, kind, 2), scalar(rnd, dom, kind, 2)))
    if pick == 1 and kind != "none":
        return Coordinate(dom if kind == "domain" else R2)
    return scalar(rnd, dom, kind, 3)


def main(trials):
    env = {u: lambda x: 1 + x[0] + 2 * x[0] * x[1], r: 0.7}
    bad = dict(value=0, static=0, grammar=0)
    rejected = 0
    for seed in range(trials):
        rnd = random.Random(seed)
        dom = rnd.choice([D, D2, Dm, Dp])
        kind = rnd.choice(["domain", "ambient", "none"])
        e = Integral(integrand(rnd, dom, kind), dom)
        out3 = integrals.expand_integrals(e)
        out = gradients.apply_gradients(pullbacks.expand_pull_backs(out3))
        rejected += not integrals.in_n4(e)
        if (shape(e), free(e)) != (shape(out), free(out)):
            bad["static"] += 1
            if bad["static"] <= 2:
                print(
                    "static type changed, seed",
                    seed,
                    kind,
                    (shape(e), free(e)),
                    (shape(out), free(out)),
                )
        bad["grammar"] += not (
            integrals.in_n4(out3)
            and integrals.in_n4(out)
            and pullbacks.in_n5(out)
            and gradients.in_n6(out)
        )
        fr = free(e)
        for combo in itertools.product(*(range(n) for n in fr.values())):
            ienv = dict(zip(fr, combo))
            before, after = evaluate(e, env, ienv)(()), evaluate(out, env, ienv)(())
            if not np.allclose(before, after, rtol=1e-6, atol=1e-8):
                bad["value"] += 1
                if bad["value"] <= 3:
                    print("value changed, seed", seed, kind, ienv, before, after)
                break
    print(
        f"{trials} integrals: {bad['value']} values changed, "
        f"{bad['static']} static types changed, "
        f"{bad['grammar']} outside N4 or N5; "
        f"N4 rejected {rejected} of {trials} inputs"
    )
    return bad | dict(lazy=trials, rejected=rejected)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 200)


def test_integrals():
    assert_clean(main(200))
