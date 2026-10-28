import random
import sys

import numpy as np
from test_integrals import R2, D
from testutil import assert_clean, contains

from uflx_oracle import evaluate
from uflx_spec_interpreter.arithmetic import add, multiply
from uflx_spec_interpreter.calculus import coordinate, gradient, integral
from uflx_spec_interpreter.domains import coordinate_domain, decomposed_domain, normal, side_map
from uflx_spec_interpreter.forms import function, space
from uflx_spec_interpreter.indices import index
from uflx_spec_interpreter.literals import num
from uflx_spec_interpreter.maps import domain_map, pull_back
from uflx_spec_interpreter.nodes import (
    CellMap,
    DomainMap,
    Function,
    Gradient,
    Normal,
    PullBack,
    SideMap,
)
from uflx_spec_interpreter.passes import gradients, integrals, pullbacks
from uflx_spec_interpreter.static import free, shape
from uflx_spec_interpreter.tensors import component_tensor, indexed, tensor
from uflx_toy.indices import index_table

I = coordinate_domain("I", 1)
k, e = index("k"), index("e")


def table(rows):
    def build(x):
        return tensor(tuple(build(y) for y in x)) if isinstance(x, (list, tuple)) else num(x)

    return build(rows)


def facets(vertices):
    # A family of straight facets with the free index e.
    fv = table(vertices)
    psi = component_tensor(
        add(
            indexed(fv, (e, 0, k)),
            multiply(
                add(indexed(fv, (e, 1, k)), multiply(num(-1), indexed(fv, (e, 0, k)))),
                indexed(coordinate(I), (0,)),
            ),
        ),
        (k,),
    )
    return decomposed_domain((domain_map(I, psi, R2),))


# The diagonal of the square, with the cell 0 on the side "+" and the cell 1 on the side "-".
E_int = facets([[(0, 0), (1, 1)]])
sides_int = {
    "+": side_map(D, E_int, index_table((0,), 2), table([[(0, 0), (0, 1)]])),
    "-": side_map(D, E_int, index_table((1,), 2), table([[(0, 0), (1, 0)]])),
}
# The four boundary edges, each with its one cell on the side "inside".
E_bnd = facets([[(0, 0), (1, 0)], [(1, 0), (1, 1)], [(1, 1), (0, 1)], [(0, 1), (0, 0)]])
sides_bnd = {
    "inside": side_map(
        D,
        E_bnd,
        index_table((0, 0, 1, 1), 2),
        table([[(0, 0), (1, 0)], [(1, 0), (0, 1)], [(1, 0), (0, 1)], [(0, 1), (0, 0)]]),
    )
}

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
                    indexed(normal(D, E, sm), (rnd.randrange(2),)),
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
        f = normal(D, E, sides[next(iter(sides))])
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
        out = gradients.apply_gradients(pullbacks.expand_pull_backs(out3))
        if contains(
            out3,
            lambda n: (
                isinstance(n, PullBack)
                and isinstance(n.domain, (CellMap, DomainMap))
                and not isinstance(n.expression, (Function, Normal, PullBack, Gradient, SideMap))
            ),
        ):
            lazy += 1
            rejected += not pullbacks.in_n5(out3)
        if (shape(e_int), free(e_int)) != (shape(out), free(out)):
            bad["static"] += 1
        bad["grammar"] += not (pullbacks.in_n5(out) and gradients.in_n6(out))
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
