import itertools
import random
import sys

import numpy as np
from testutil import assert_clean

from uflx_spec_interpreter.arithmetic import add, multiply
from uflx_spec_interpreter.calculus import coordinate, gradient, integral
from uflx_spec_interpreter.domains import coordinate_domain, decomposed_domain
from uflx_spec_interpreter.forms import function, space
from uflx_spec_interpreter.indices import index
from uflx_spec_interpreter.literals import num
from uflx_spec_interpreter.maps import domain_map
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import gradients, integrals, pullbacks
from uflx_spec_interpreter.static import free, shape
from uflx_spec_interpreter.tensors import component_tensor, indexed, tensor

R2 = coordinate_domain("R2", 2)
K = coordinate_domain("K", 2)
X = coordinate(K)
k = index("k")
c = index("c")


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

# The square as one family of two triangles, with the free index c.
p = tensor(tuple(tensor(tuple(tensor((num(a), num(b))) for a, b in tri)) for tri in (T0, T1)))
fam = lambda i: indexed(p, (c, i, k))
psi_c = component_tensor(
    add(
        fam(0),
        add(
            multiply(add(fam(1), multiply(num(-1), fam(0))), indexed(X, (0,))),
            multiply(add(fam(2), multiply(num(-1), fam(0))), indexed(X, (1,))),
        ),
    ),
    (k,),
)
D = decomposed_domain((domain_map(K, psi_c, R2),))

# A family of three points in the square.
Dp = decomposed_domain(
    (
        domain_map(
            coordinate_domain("pt", 0),
            component_tensor(
                indexed(
                    tensor(
                        tuple(
                            tensor((num(a), num(b)))
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
D2 = decomposed_domain((domain_map(K, affine(T0, X), R2), domain_map(K, affine(T1, X), R2)))

# A macro triangle: T0 as the image of the unit triangle split into two sub-triangles.
Kdec = decomposed_domain(
    (
        domain_map(K, affine([(0, 0), (1, 0), (0.5, 0.5)], X), K),
        domain_map(K, affine([(0, 0), (0.5, 0.5), (0, 1)], X), K),
    )
)
Dm = decomposed_domain((domain_map(Kdec, affine(T0, coordinate(Kdec)), R2),))

V = space("V", domain=D)
u = function(V, "u")
r = function(space("R"), "r")
d = tensor((num(3), num(5)))


def scalar(rnd, dom, kind, depth):
    if depth == 0 or rnd.random() < 0.35:
        leaves = {
            "domain": [indexed(coordinate(dom), (0,)), indexed(coordinate(dom), (1,))]
            + ([u, indexed(gradient(u, D), (rnd.randrange(2),))] if dom == D else []),
            "ambient": [indexed(coordinate(R2), (0,)), indexed(coordinate(R2), (1,))]
            + ([indexed(d, (c,))] if dom == D else []),
            "none": [num(rnd.choice([1, 2, -0.5])), r],
        }[kind]
        return rnd.choice([*leaves, num(2)])
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
        dom = rnd.choice([D, D2, Dm, Dp])
        kind = rnd.choice(["domain", "ambient", "none"])
        e = integral(integrand(rnd, dom, kind), dom)
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
