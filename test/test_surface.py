import random
import sys

import numpy as np
from test_integrals import BOUNDARY, DIAGONAL, SQUARE
from testutil import assert_clean, check_types, contains, keeps_static_type

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    RD,
    add,
    component_tensor,
    coordinate,
    expand_arithmetic,
    function,
    gradient,
    image,
    index,
    index_sum,
    indexed,
    integral,
    multiply,
    normal,
    num,
    pull_back,
    side_domain,
    side_map,
    tensor,
)
from uflx_spec_interpreter._src.nodes import (
    CellMap,
    Coordinate,
    Function,
    Gradient,
    Normal,
    PullBack,
)
from uflx_spec_interpreter._src.passes import integrals
from uflx_spec_interpreter._src.passes.contracts import _in_core_normal_form
from uflx_spec_interpreter._src.passes.lowering import _lower_spatial
from uflx_spec_interpreter._src.passes.pullbacks import _pull_backs_expanded
from uflx_toy.cells import reference_simplex
from uflx_toy.nodes import VertexCoordinates
from uflx_toy.spaces import space

R3 = RD(3)
I = reference_simplex(1)
k, i = index("k"), index("i")

# A surface folded along its diagonal: two triangles that are not coplanar.
FOLD = VertexCoordinates(
    SQUARE, ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 1.0, 0.5), (0.0, 1.0, 1.0))
)
M = image(FOLD, R3)

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


# The same curved surface written as a mesh, with the cells of M deformed, so it has sides.
C = image(deform(FOLD), R3)


def sides(D):
    # The fold and the boundary of the surface D, from the topologies of the square.
    E_int, E_bnd = (
        side_domain(D, DIAGONAL, DIAGONAL.sides[0][0][0]),
        side_domain(D, BOUNDARY, BOUNDARY.sides[0][0][0]),
    )
    interior = {s: side_map(D, E_int, s) for s in ("+", "-")}
    return [(E_int, interior), (E_bnd, {"inside": side_map(D, E_bnd, "inside")})]


EDGES = {M: sides(M), C: sides(C)}

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
    n = normal(sm)
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
    out = _lower_spatial(out3)
    out = expand_arithmetic(out)
    if contains(
        out3,
        lambda n: (
            isinstance(n, PullBack)
            and isinstance(n.map, CellMap)
            and not isinstance(n.expression, (Function, Normal, Coordinate, PullBack, Gradient))
        ),
    ):
        bad["lazy"] += 1
        bad["rejected"] += not _pull_backs_expanded(out3)
    check_types(bad, e, out)
    if not keeps_static_type(e, out):
        bad["static"] += 1
    bad["grammar"] += not (_pull_backs_expanded(out) and _in_core_normal_form(out))
    before, after = evaluate(e, env)(()), evaluate(out, env)(())
    if not np.allclose(before, after, rtol=1e-4, atol=1e-6):
        bad["value"] += 1
        if bad["value"] <= 3:
            print("value changed:", name, before, after)


def main(trials):
    bad = dict(value=0, static=0, grammar=0, lazy=0, rejected=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        kind = rnd.choice(["surface", "curved surface", "edges", "curved edges"])
        if kind == "surface":
            e_int = integral(field(rnd, M), M)
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
