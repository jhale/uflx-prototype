import itertools
import random
import sys

import numpy as np

from uflx._src.passes import integrals
from uflx._src.passes.contracts import in_core_normal_form
from uflx._src.passes.integrals import integrals_expanded
from uflx._src.passes.lowering import lower_spatial
from uflx._src.passes.pullbacks import pull_backs_expanded
from uflx.arithmetic import add, multiply, num
from uflx.calculus import coordinate, gradient, integral
from uflx.forms import function
from uflx.nodes import EuclideanSpace
from uflx.passes import expand_arithmetic
from uflx.tensors import component_tensor, index, indexed, tensor
from uflx_basix.cells import reference_simplex
from uflx_basix.spaces import space
from uflx_oracle import evaluate

from ..testutil import assert_clean, check_types, grouped_topology, keeps_static_type, vertex_mesh

R2 = EuclideanSpace(2)
K = reference_simplex(2)
X = coordinate(K)
k = index("k")
# The cells of the square, which the folded surface of test_surface shares.
SQUARE = grouped_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
I = reference_simplex(1)
# The diagonal, seen from both triangles, and the boundary, seen from inside.
DIAGONAL = grouped_topology((I,), (((0, 2),),), ((("+", SQUARE, 0), ("-", SQUARE, 0)),))
BOUNDARY = grouped_topology((I,), (((0, 1), (1, 2), (2, 3), (3, 0)),), ((("inside", SQUARE, 0),),))
c = index("load-entry")


def affine(vertices, at):
    # The affine map of the unit triangle onto a triangle, as an expression of the coordinate at.
    v0, v1, v2 = (tensor(tuple(num(x) for x in v)) for v in vertices)
    term = lambda v: indexed(v, (k,))
    body = add(
        term(v0),
        add(
            multiply(add(term(v1), multiply(num(-1), term(v0))), indexed(at, (0,))),
            multiply(add(term(v2), multiply(num(-1), term(v0))), indexed(at, (1,))),
        ),
    )
    return component_tensor(body, (k,))


T0, T1 = [(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]

# The square as a mesh of the coordinates of its vertices, with the cell index c of SQUARE.
D = vertex_mesh(SQUARE, [(0, 0), (1, 0), (1, 1), (0, 1)])

# Point fields get an explicit topology of independent zero-dimensional cells.
POINTS = grouped_topology((reference_simplex(0),), (((0,), (1,), (2,)),), ())
Dp = vertex_mesh(POINTS, [(0.2, 0.3), (0.5, 0.5), (0.9, 0.1)])

# Two independent triangle patches: no shared vertices or sides are required.
PATCHES = grouped_topology((K,), (((0, 1, 2), (3, 4, 5)),), ())
D2 = vertex_mesh(PATCHES, [*T0, *T1])

V = space("V", domain=D)
u = function(V, "u")
r = function(space("R"), "r")
d = tensor((num(3), num(5)))


def leaves(rnd, dom, kind):
    match kind:
        case "domain":
            return [indexed(coordinate(dom), (0,)), indexed(coordinate(dom), (1,))] + (
                [u, indexed(gradient(u, D), (rnd.randrange(2),))] if dom == D else []
            )
        case "ambient":
            return [indexed(coordinate(R2), (0,)), indexed(coordinate(R2), (1,))] + (
                [indexed(d, (c,))] if dom == D else []
            )
    return [num(rnd.choice([1, 2, -0.5])), r]


def scalar(rnd, dom, kind, depth):
    if depth == 0 or rnd.random() < 0.35:
        return rnd.choice([*leaves(rnd, dom, kind), num(2)])
    a, b = scalar(rnd, dom, kind, depth - 1), scalar(rnd, dom, kind, depth - 1)
    return rnd.choice([add(a, b), multiply(a, b)])


def integrand(rnd, dom, kind):
    pick = rnd.randrange(4)
    if pick == 0:
        return tensor((scalar(rnd, dom, kind, 2), scalar(rnd, dom, kind, 2)))
    if pick == 1 and kind != "none":
        return coordinate(dom if kind == "domain" else R2)
    return scalar(rnd, dom, kind, 3)


def main(trials):
    env = {u: lambda x: 1 + x[0] + 2 * x[0] * x[1], r: 0.7}
    bad = dict(value=0, static=0, grammar=0)
    rejected = 0
    for seed in range(trials):
        rnd = random.Random(seed)
        dom = rnd.choice([D, D2, Dp])
        kind = rnd.choice(["domain", "ambient", "none"])
        e = integral(integrand(rnd, dom, kind), dom)
        out3 = integrals.expand_integrals(e)
        out = expand_arithmetic(lower_spatial(out3))
        rejected += not integrals_expanded(e)
        check_types(bad, e, out)
        if not keeps_static_type(e, out):
            bad["static"] += 1
            if bad["static"] <= 2:
                print(
                    "static type changed, seed",
                    seed,
                    kind,
                    (e.value_shape, dict(e.free_indices)),
                    (out.value_shape, dict(out.free_indices)),
                )
        bad["grammar"] += not (
            integrals_expanded(out3)
            and integrals_expanded(out)
            and pull_backs_expanded(out)
            and in_core_normal_form(out)
        )
        fr = dict(e.free_indices)
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
