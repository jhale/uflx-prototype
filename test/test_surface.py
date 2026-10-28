import random
import sys

import numpy as np
from test_facets import table
from test_integrals import K
from testutil import assert_clean, check_types, contains, static_type

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    RD,
    add,
    component_tensor,
    coordinate,
    decomposed_domain,
    domain_map,
    function,
    gradient,
    index,
    index_sum,
    indexed,
    integral,
    multiply,
    normal,
    num,
    pull_back,
    side_map,
    tensor,
)
from uflx_spec_interpreter._src.nodes import (
    CellMap,
    DomainMap,
    Function,
    Gradient,
    Normal,
    PullBack,
    SideMap,
)
from uflx_spec_interpreter._src.passes import gradients, integrals, pullbacks
from uflx_toy.cells import reference_simplex
from uflx_toy.indices import index_table
from uflx_toy.spaces import space

R3 = RD(3)
I = reference_simplex(1)
k, c, e, i = index("k"), index("c"), index("e"), index("i")

# A surface folded along its diagonal: two triangles that are not coplanar.
p = table([[(0, 0, 0), (1, 0, 0), (1, 1, 0.5)], [(0, 0, 0), (1, 1, 0.5), (0, 1, 1)]])
fam = lambda v: indexed(p, (c, v, k))
X = coordinate(K)
psi = component_tensor(
    add(
        fam(0),
        add(
            multiply(add(fam(1), multiply(num(-1), fam(0))), indexed(X, (0,))),
            multiply(add(fam(2), multiply(num(-1), fam(0))), indexed(X, (1,))),
        ),
    ),
    (k,),
)
M = decomposed_domain((domain_map(K, psi, R3),))

bump = tensor((num(1), num(-1), num(2)))
kk = index("kk")


def deform(z):
    # A smooth map of R^3 applied to the point z.
    return component_tensor(
        add(
            indexed(z, (kk,)),
            multiply(
                multiply(num(0.2), multiply(indexed(z, (0,)), indexed(z, (1,)))),
                indexed(bump, (kk,)),
            ),
        ),
        (kk,),
    )


# The deformed surface on the parameter domain M, which has no topology and so no sides.
Ms = decomposed_domain((domain_map(M, deform(coordinate(M)), R3),))

# The same curved surface written as a mesh, with the cells of M deformed, so it has sides.
C = decomposed_domain((domain_map(K, deform(psi), R3),))


def edges(vertices, curved=False):
    fv = table(vertices)
    psi_e = component_tensor(
        add(
            indexed(fv, (e, 0, k)),
            multiply(
                add(indexed(fv, (e, 1, k)), multiply(num(-1), indexed(fv, (e, 0, k)))),
                indexed(coordinate(I), (0,)),
            ),
        ),
        (k,),
    )
    return decomposed_domain((domain_map(I, deform(psi_e) if curved else psi_e, R3),))


INTERIOR = [[(0, 0, 0), (1, 1, 0.5)]]
BOUNDARY = [
    [(0, 0, 0), (1, 0, 0)],
    [(1, 0, 0), (1, 1, 0.5)],
    [(1, 1, 0.5), (0, 1, 1)],
    [(0, 1, 1), (0, 0, 0)],
]


def sides(D, curved):
    # The fold and the boundary of the surface D, with their side tables.
    E_int, E_bnd = edges(INTERIOR, curved), edges(BOUNDARY, curved)
    interior = {
        "+": side_map(D, E_int, index_table((0,), 2), table([[(0, 0), (0, 1)]])),
        "-": side_map(D, E_int, index_table((1,), 2), table([[(0, 0), (1, 0)]])),
    }
    boundary = {
        "inside": side_map(
            D,
            E_bnd,
            index_table((0, 0, 1, 1), 2),
            table([[(0, 0), (1, 0)], [(1, 0), (0, 1)], [(1, 0), (0, 1)], [(0, 1), (0, 0)]]),
        )
    }
    return [(E_int, interior), (E_bnd, boundary)]


EDGES = {M: sides(M, False), C: sides(C, True)}

u = function(space("V", domain=M), "u")
ENV = {u: {0: lambda x: 1 + x[0] + 2 * x[1] - x[2], 1: lambda x: 3 - x[0] * x[2] + x[1]}}


def field(rnd, dom):
    Z = coordinate(dom)
    g = add(multiply(indexed(Z, (0,)), indexed(Z, (1,))), indexed(Z, (2,)))
    h = multiply(indexed(Z, (2,)), indexed(Z, (2,)))
    grad = lambda q: gradient(q, dom)
    return rnd.choice(
        [
            g,
            indexed(grad(g), (rnd.randrange(3),)),
            index_sum(multiply(indexed(grad(g), (i,)), indexed(grad(h), (i,))), (i,)),
            multiply(h, indexed(grad(h), (rnd.randrange(3),))),
            num(2),
        ]
    )


def edge_integrand(rnd, D, E, sides):
    sm = sides[rnd.choice(list(sides))]
    Z = coordinate(D)
    g = add(multiply(indexed(Z, (0,)), indexed(Z, (1,))), indexed(Z, (2,)))
    f = rnd.choice(
        ([u, indexed(gradient(u, M), (rnd.randrange(3),))] if D == M else [])
        + [
            indexed(Z, (rnd.randrange(3),)),
            indexed(gradient(g, D), (rnd.randrange(3),)),
        ]
    )
    n = normal(D, E, sm)
    return rnd.choice(
        [
            pull_back(f, sm),
            multiply(pull_back(f, sm), indexed(n, (rnd.randrange(3),))),
            n,
            add(
                pull_back(f, sides[next(iter(sides))]),
                multiply(num(-1), pull_back(f, sides[list(sides)[-1]])),
            ),
        ]
    )


def check(name, e, env, bad, rows):
    out3 = integrals.expand_integrals(e)
    out = gradients.apply_gradients(pullbacks.expand_pull_backs(out3))
    if contains(
        out3,
        lambda n: (
            isinstance(n, PullBack)
            and isinstance(n.domain, (CellMap, DomainMap))
            and not isinstance(n.expression, (Function, Normal, PullBack, Gradient, SideMap))
        ),
    ):
        bad["lazy"] += 1
        bad["rejected"] += not pullbacks.in_n5(out3)
    check_types(bad, e, out)
    if static_type(e) != static_type(out):
        bad["static"] += 1
    bad["grammar"] += not (pullbacks.in_n5(out) and gradients.in_n6(out))
    before, after = evaluate(e, env)(()), evaluate(out, env)(())
    if not np.allclose(before, after, rtol=1e-4, atol=1e-6):
        bad["value"] += 1
        if bad["value"] <= 3:
            print("value changed:", name, before, after)


def main(trials):
    bad = dict(value=0, static=0, grammar=0, lazy=0, rejected=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        kind = rnd.choice(
            ["surface", "deformed surface", "curved surface", "edges", "curved edges"]
        )
        if kind == "surface":
            e_int = integral(field(rnd, M), M)
        elif kind == "deformed surface":
            e_int = integral(field(rnd, Ms), Ms)
        elif kind == "curved surface":
            e_int = integral(field(rnd, C), C)
        else:
            D = C if kind == "curved edges" else M
            E, side_maps = rnd.choice(EDGES[D])
            e_int = integral(edge_integrand(rnd, D, E, side_maps), E)
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
