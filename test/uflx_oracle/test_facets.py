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
from uflx.geometry import pull_back
from uflx.passes import expand_arithmetic
from uflx.tensors import index, indexed
from uflx_basix.cells import reference_simplex
from uflx_basix.sides import outward_conormal
from uflx_basix.spaces import space
from uflx_oracle.interpreter import Interpreter

from ..testutil import assert_clean, check_types, contains, keeps_static_type
from ..uflx_basix.integrals import DIAGONAL, D
from ..uflx_basix.meshes import side_domain, side_map
from .test_integrals import BOUNDARY

I = reference_simplex(1)
k, e = index("k"), index("e")


# The diagonal and the boundary of the square, as meshes of their topologies.
E_int = side_domain(D, DIAGONAL)
sides_int = {s: side_map(D, E_int, s) for s in ("+", "-")}
E_bnd = side_domain(D, BOUNDARY)
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
                    indexed(outward_conormal(sm), (rnd.randrange(2),)),
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
        f = outward_conormal(sides[next(iter(sides))])
    return f


def main(trials):
    x = coordinate(D)
    env = {
        u: {0: num(1) + x[0] + num(2) * x[1], 1: num(3) - x[0] * x[1]},
        w: num(2) + x[0] * x[1],
    }
    bad = dict(value=0, static=0, grammar=0)
    lazy, rejected = 0, 0
    for seed in range(trials):
        rnd = random.Random(seed)
        E, sides = rnd.choice([(E_int, sides_int), (E_bnd, sides_bnd)])
        e_int = integral(integrand(rnd, E, sides), E)
        out3 = integrals.expand_integrals(e_int)
        out = lower_spatial(out3)
        out = expand_arithmetic(out)
        if contains(
            out3,
            lambda n: (
                isinstance(n, PullBack)
                and isinstance(n.map, CellMap)
                and not isinstance(
                    n.expression, (Function, Conormal, Coordinate, PullBack, Gradient)
                )
            ),
        ):
            lazy += 1
            rejected += not pull_backs_expanded(out3)
        check_types(bad, e_int, out)
        if not keeps_static_type(e_int, out):
            bad["static"] += 1
        bad["grammar"] += not (pull_backs_expanded(out) and in_core_normal_form(out))
        oracle = Interpreter(env)
        before, after = oracle.evaluate(e_int), oracle.evaluate(out)
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
