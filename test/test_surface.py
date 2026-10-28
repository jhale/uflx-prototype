import random
import sys

import numpy as np
from test_facets import table
from test_integrals import K
from testutil import assert_clean, contains

from uflx_spec_interpreter import integrals, pullbacks
from uflx_spec_interpreter.core import (
    CT,
    CellMap,
    CoordDomain,
    Coordinate,
    Decomposed,
    DomainMap,
    Fn,
    Gradient,
    Index,
    Indexed,
    Integral,
    ISum,
    Normal,
    Num,
    Prod,
    PullBack,
    SideMap,
    Space,
    Sum,
    Tensor,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.static import free, shape

R3 = CoordDomain("R3", 3)
I = CoordDomain("I", 1)
k, c, e, i = Index("k"), Index("c"), Index("e"), Index("i")

# A surface folded along its diagonal: two triangles that are not coplanar.
p = table([[(0, 0, 0), (1, 0, 0), (1, 1, 0.5)], [(0, 0, 0), (1, 1, 0.5), (0, 1, 1)]])
fam = lambda v: Indexed(p, (c, v, k))
X = Coordinate(K)
psi = CT(
    Sum(
        fam(0),
        Sum(
            Prod(Sum(fam(1), Prod(Num(-1), fam(0))), Indexed(X, (0,))),
            Prod(Sum(fam(2), Prod(Num(-1), fam(0))), Indexed(X, (1,))),
        ),
    ),
    (k,),
)
M = Decomposed((DomainMap(K, psi, R3),))

# The surface deformed by a smooth map of R^3, on the parameter domain M of dimension 2.
Y = Coordinate(M)
bump = Tensor((Num(1), Num(-1), Num(2)))
phi = CT(
    Sum(
        Indexed(Y, (k,)),
        Prod(Prod(Num(0.2), Prod(Indexed(Y, (0,)), Indexed(Y, (1,)))), Indexed(bump, (k,))),
    ),
    (k,),
)
Ms = Decomposed((DomainMap(M, phi, R3),))


def edges(vertices):
    fv = table(vertices)
    psi_e = CT(
        Sum(
            Indexed(fv, (e, 0, k)),
            Prod(
                Sum(Indexed(fv, (e, 1, k)), Prod(Num(-1), Indexed(fv, (e, 0, k)))),
                Indexed(Coordinate(I), (0,)),
            ),
        ),
        (k,),
    )
    return Decomposed((DomainMap(I, psi_e, R3),))


E_int = edges([[(0, 0, 0), (1, 1, 0.5)]])
sides_int = {
    "+": SideMap(M, E_int, table([0]), table([[(0, 0), (0, 1)]])),
    "-": SideMap(M, E_int, table([1]), table([[(0, 0), (1, 0)]])),
}
E_bnd = edges(
    [
        [(0, 0, 0), (1, 0, 0)],
        [(1, 0, 0), (1, 1, 0.5)],
        [(1, 1, 0.5), (0, 1, 1)],
        [(0, 1, 1), (0, 0, 0)],
    ]
)
sides_bnd = {
    "inside": SideMap(
        M,
        E_bnd,
        table([0, 0, 1, 1]),
        table([[(0, 0), (1, 0)], [(1, 0), (0, 1)], [(1, 0), (0, 1)], [(0, 1), (0, 0)]]),
    )
}

u = Fn(Space("V", domain=M), "u")
ENV = {u: {0: lambda x: 1 + x[0] + 2 * x[1] - x[2], 1: lambda x: 3 - x[0] * x[2] + x[1]}}


def field(rnd, dom):
    Z = Coordinate(dom)
    g = Sum(Prod(Indexed(Z, (0,)), Indexed(Z, (1,))), Indexed(Z, (2,)))
    h = Prod(Indexed(Z, (2,)), Indexed(Z, (2,)))
    grad = lambda q: Gradient(q, dom)
    return rnd.choice(
        [
            g,
            Indexed(grad(g), (rnd.randrange(3),)),
            ISum(Prod(Indexed(grad(g), (i,)), Indexed(grad(h), (i,))), (i,)),
            Prod(h, Indexed(grad(h), (rnd.randrange(3),))),
            Num(2),
        ]
    )


def edge_integrand(rnd, E, sides):
    sm = sides[rnd.choice(list(sides))]
    f = rnd.choice(
        [
            u,
            Indexed(Gradient(u, M), (rnd.randrange(3),)),
            Indexed(Coordinate(M), (rnd.randrange(3),)),
        ]
    )
    n = Normal(M, E, sm)
    return rnd.choice(
        [
            PullBack(f, sm),
            Prod(PullBack(f, sm), Indexed(n, (rnd.randrange(3),))),
            n,
            Sum(
                PullBack(u, sides[next(iter(sides))]),
                Prod(Num(-1), PullBack(u, sides[list(sides)[-1]])),
            ),
        ]
    )


def check(name, e, env, bad, rows):
    out3 = integrals.expand_integrals(e)
    out = pullbacks.expand_pull_backs(out3)
    if contains(
        out3,
        lambda n: (
            isinstance(n, PullBack)
            and isinstance(n.m, (CellMap, DomainMap))
            and not isinstance(n.b, (Fn, Normal, PullBack, Gradient, SideMap))
        ),
    ):
        bad["lazy"] += 1
        bad["rejected"] += not pullbacks.in_n5(out3)
    if (shape(e), free(e)) != (shape(out), free(out)):
        bad["static"] += 1
    bad["grammar"] += not pullbacks.in_n5(out)
    before, after = evaluate(e, env)(()), evaluate(out, env)(())
    if not np.allclose(before, after, rtol=1e-4, atol=1e-6):
        bad["value"] += 1
        if bad["value"] <= 3:
            print("value changed:", name, before, after)


def main(trials):
    bad = dict(value=0, static=0, grammar=0, lazy=0, rejected=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        kind = rnd.choice(["surface", "deformed surface", "edges"])
        if kind == "surface":
            e_int = Integral(field(rnd, M), M)
        elif kind == "deformed surface":
            e_int = Integral(field(rnd, Ms), Ms)
        else:
            E, sides = rnd.choice([(E_int, sides_int), (E_bnd, sides_bnd)])
            e_int = Integral(edge_integrand(rnd, E, sides), E)
        check(f"{kind} {seed}", e_int, ENV, bad, None)
    print(
        f"{trials} integrals on a surface in R^3: {bad['value']} values changed, "
        f"{bad['static']} static types changed, "
        f"{bad['grammar']} outside N5; "
        f"N5 rejected {bad['rejected']} of {bad['lazy']} inputs with a pull-back to expand"
    )
    return bad


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 30)


def test_surface():
    assert_clean(main(60))
