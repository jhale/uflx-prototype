import random
import sys

import numpy as np

from uflx._src.nodes import CellMap, Conormal, Coordinate, Function, Gradient, PullBack
from uflx._src.passes import integrals
from uflx._src.passes.contracts import in_core_normal_form
from uflx._src.passes.lowering import lower_spatial
from uflx._src.passes.pullbacks import pull_backs_expanded
from uflx.arithmetic import add, multiply, num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import function
from uflx.geometry import mesh, pull_back
from uflx.passes import expand_arithmetic
from uflx.tensors import component_tensor, index, index_sum, indexed, tensor
from uflx_basix.cells import reference_simplex
from uflx_basix.sides import outward_conormal
from uflx_basix.spaces import space
from uflx_oracle.interpreter import Interpreter

from ..testutil import assert_clean, check_types, contains, keeps_static_type
from ..uflx_basix.integrals import DIAGONAL
from ..uflx_basix.meshes import side_domain, side_map
from ..uflx_basix.surface import FOLD, R3, M
from .test_integrals import BOUNDARY

I = reference_simplex(1)
k, i = index("k"), index("i")

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
C = mesh(deform(FOLD), R3)


def sides(D):
    # The fold and the boundary of the surface D, from the topologies of the square.
    E_int, E_bnd = (
        side_domain(D, DIAGONAL),
        side_domain(D, BOUNDARY),
    )
    interior = {s: side_map(D, E_int, s) for s in ("+", "-")}
    return [(E_int, interior), (E_bnd, {"inside": side_map(D, E_bnd, "inside")})]


EDGES = {M: sides(M), C: sides(C)}

u = function(space("V", domain=M), "u")
x = coordinate(M)
ENV = {u: {0: num(1) + x[0] + num(2) * x[1] - x[2], 1: num(3) - x[0] * x[2] + x[1]}}


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
    n = outward_conormal(sm)
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
    out = lower_spatial(out3)
    out = expand_arithmetic(out)
    if contains(
        out3,
        lambda n: (
            isinstance(n, PullBack)
            and isinstance(n.map, CellMap)
            and not isinstance(n.expression, (Function, Conormal, Coordinate, PullBack, Gradient))
        ),
    ):
        bad["lazy"] += 1
        bad["rejected"] += not pull_backs_expanded(out3)
    check_types(bad, e, out)
    if not keeps_static_type(e, out):
        bad["static"] += 1
    bad["grammar"] += not (pull_backs_expanded(out) and in_core_normal_form(out))
    oracle = Interpreter(env)
    before, after = oracle.evaluate(e), oracle.evaluate(out)
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
