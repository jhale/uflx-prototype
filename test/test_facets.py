import random
import sys

import numpy as np
from test_integrals import BOUNDARY, DIAGONAL, D
from testutil import assert_clean, check_types, contains, keeps_static_type

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    add,
    coordinate,
    expand_arithmetic,
    function,
    gradient,
    index,
    indexed,
    integral,
    multiply,
    normal,
    num,
    pull_back,
    side_domain,
    side_map,
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
from uflx_toy.spaces import space

I = reference_simplex(1)
k, e = index("k"), index("e")


# The diagonal and the boundary of the square, as meshes of their topologies.
E_int = side_domain(D, DIAGONAL, DIAGONAL.sides[0][0][0])
sides_int = {s: side_map(D, E_int, s) for s in ("+", "-")}
E_bnd = side_domain(D, BOUNDARY, BOUNDARY.sides[0][0][0])
sides_bnd = {"inside": side_map(D, E_bnd, "inside")}

V = space("V", domain=D)
u, w = function(V, "u"), function(V, "w")


def restricted(rnd, sm):
    f = rnd.choice(
        [
            u,
            w,
            indexed(gradient(u, D), (rnd.randrange(2),)),
            indexed(coordinate(D), (rnd.randrange(2),)),
            multiply(u, w),
            indexed(gradient(multiply(u, w), D), (rnd.randrange(2),)),
        ]
    )
    return pull_back(f, sm)


def integrand(rnd, E, sides):
    terms = []
    for _ in range(rnd.randrange(1, 3)):
        sm = sides[rnd.choice(list(sides))]
        pick = rnd.randrange(4)
        if pick == 0:
            terms.append(restricted(rnd, sm))
        elif pick == 1:
            terms.append(
                multiply(
                    restricted(rnd, sm),
                    indexed(normal(sm), (rnd.randrange(2),)),
                )
            )
        elif pick == 2:
            a, b = sides[next(iter(sides))], sides[list(sides)[-1]]
            terms.append(add(restricted(rnd, a), multiply(num(-1), restricted(rnd, b))))
        else:
            terms.append(num(rnd.choice([1, 2])))
    f = terms[0]
    for t in terms[1:]:
        f = add(f, t)
    if rnd.random() < 0.2:
        f = normal(sides[next(iter(sides))])
    return f


def main(trials):
    env = {
        u: {0: lambda x: 1 + x[0] + 2 * x[1], 1: lambda x: 3 - x[0] * x[1]},
        w: lambda x: 2 + x[0] * x[1],
    }
    bad = dict(value=0, static=0, grammar=0)
    lazy, rejected = 0, 0
    for seed in range(trials):
        rnd = random.Random(seed)
        E, sides = rnd.choice([(E_int, sides_int), (E_bnd, sides_bnd)])
        e_int = integral(integrand(rnd, E, sides), E)
        out3 = integrals.expand_integrals(e_int)
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
            lazy += 1
            rejected += not _pull_backs_expanded(out3)
        check_types(bad, e_int, out)
        if not keeps_static_type(e_int, out):
            bad["static"] += 1
        bad["grammar"] += not (_pull_backs_expanded(out) and _in_core_normal_form(out))
        before, after = evaluate(e_int, env)(()), evaluate(out, env)(())
        if not np.allclose(before, after, rtol=1e-5, atol=1e-7):
            bad["value"] += 1
            if bad["value"] <= 3:
                print("value changed, seed", seed, before, after)
    print(
        f"{trials} facet integrals: {bad['value']} values changed, "
        f"{bad['static']} static types changed, "
        f"{bad['grammar']} outside N5; "
        f"N5 rejected {rejected} of {lazy} inputs with a pull-back to expand"
    )
    return bad | dict(lazy=lazy, rejected=rejected)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)


def test_facets():
    assert_clean(main(60))
