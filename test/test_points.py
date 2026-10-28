import random
import sys

import numpy as np
from test_facets import table
from test_integrals import R2, D, u
from test_surface import R3, M
from testutil import assert_clean, contains

from uflx_spec_interpreter.algebra import (
    add,
    component_tensor,
    index,
    index_table,
    indexed,
    multiply,
    num,
)
from uflx_spec_interpreter.calculus import coordinate, gradient, integral
from uflx_spec_interpreter.domains import coordinate_domain, decomposed_domain, side_map
from uflx_spec_interpreter.forms import function, space
from uflx_spec_interpreter.maps import domain_map, pull_back
from uflx_spec_interpreter.nodes import (
    Coordinate,
    DomainMap,
    Function,
    Gradient,
    PullBack,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import gradients, integrals, pullbacks
from uflx_spec_interpreter.static import free, shape

i, k = index("i"), index("k")
POINT = coordinate_domain("pt", 0)


def points(coordinates, Q):
    # A family of points in Q, with the free index i.
    return decomposed_domain(
        (domain_map(POINT, component_tensor(indexed(table(coordinates), (i, k)), (k,)), Q),)
    )


def on_surface(cell, X):
    # A point of the folded surface M, from a reference point of one of its two triangles.
    verts = [[(0, 0, 0), (1, 0, 0), (1, 1, 0.5)], [(0, 0, 0), (1, 1, 0.5), (0, 1, 1)]][cell]
    v0, v1, v2 = (np.array(v, dtype=float) for v in verts)
    return tuple(float(c) for c in v0 + (v1 - v0) * X[0] + (v2 - v0) * X[1])


SQUARE = [(0.2, 0.3), (0.7, 0.4), (0.35, 0.8), (0.9, 0.95)]
SURFACE = [(0, (0.3, 0.2)), (1, (0.25, 0.5))]
SQUARE_POINTS = points(SQUARE, D)
SURFACE_POINTS = points([on_surface(c, X) for c, X in SURFACE], M)


def in_square(x):
    # The cell of the square that contains x, and the reference point, as a mesh library finds them.
    for cell, (v0, v1, v2) in enumerate([[(0, 0), (1, 0), (1, 1)], [(0, 0), (1, 1), (0, 1)]]):
        T = np.array([np.subtract(v1, v0), np.subtract(v2, v0)]).T
        X = np.linalg.solve(T, np.subtract(x, v0))
        if np.all(X > -1e-12) and X.sum() < 1 + 1e-12:
            return cell, tuple(float(c) for c in X)
    raise ValueError(x)


def topology_of_points(Q, coordinates, located):
    # The points as a topology with the side "inside", and the restriction to them.
    E = points(coordinates, R2 if Q == D else R3)
    cells = index_table(tuple(cell for cell, _ in located), 2)
    positions = table([[X] for _, X in located])
    return E, side_map(Q, E, cells, positions)


SQUARE_SIDE = topology_of_points(D, SQUARE, [in_square(x) for x in SQUARE])
SURFACE_SIDE = topology_of_points(M, [on_surface(c, X) for c, X in SURFACE], SURFACE)
w = function(space("W", domain=M), "w")
ENV = {u: lambda x: 1 + x[0] + 2 * x[0] * x[1], w: lambda x: 1 + x[0] * x[1] + x[2] * x[2]}


def integrand(rnd, Q, f):
    X = coordinate(Q)
    n = 2 if Q == D else 3
    r, s = rnd.randrange(n), rnd.randrange(n)
    grad = lambda g: gradient(g, Q)
    choices = [
        f,
        indexed(X, (r,)),
        indexed(grad(f), (r,)),
        indexed(grad(multiply(f, f)), (r,)),
        indexed(grad(multiply(f, indexed(X, (0,)))), (r,)),
        indexed(grad(X), (r, s)),
    ]
    if Q == D:
        # Hessians of a terminal and of a product, both evaluated on the mesh.
        choices += [indexed(grad(grad(f)), (r, s)), indexed(grad(grad(multiply(f, f))), (r, s))]
    a, b = rnd.choice(choices), rnd.choice(choices)
    return rnd.choice([a, add(a, b), multiply(a, b), multiply(num(2), a)])


def stays_for_the_mesh_library(n):
    # A pull-back by the points of a function or of its gradient remains for the mesh library.
    return isinstance(n, PullBack) and isinstance(n.expression, (Function, Gradient))


def needs_pass(n):
    return isinstance(n, Gradient) and not isinstance(
        n.expression, (Function, Coordinate, Gradient, PullBack)
    )


def by_the_points(n):
    return (
        isinstance(n, PullBack)
        and isinstance(n.domain, DomainMap)
        and n.domain.ambient_domain in (D, M)
    )


def main(trials):
    bad = dict(value=0, static=0, grammar=0, lazy=0, rejected=0)
    bad |= dict(unlocated=0, unlowered=0, restricted=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        Q, family, side, f = rnd.choice(
            [
                (D, SQUARE_POINTS, SQUARE_SIDE, u),
                (M, SURFACE_POINTS, SURFACE_SIDE, w),
            ]
        )
        restricted = rnd.random() < 0.5
        if restricted:
            E, sm = side
            e = integral(pull_back(integrand(rnd, Q, f), sm), E)
        else:
            e = integral(integrand(rnd, Q, f), family)
        out4 = pullbacks.expand_pull_backs(integrals.expand_integrals(e))
        out = gradients.apply_gradients(out4)
        if contains(out4, needs_pass):
            bad["lazy"] += 1
            bad["rejected"] += not gradients.in_n6(out4)
        if (shape(e), free(e)) != (shape(out), free(out)):
            bad["static"] += 1
        bad["grammar"] += not (pullbacks.in_n5(out) and gradients.in_n6(out))
        if restricted:
            # Through the restriction, every pull-back reads a library value on a cell.
            bad["restricted"] += 1
            bad["unlowered"] += contains(out, by_the_points)
        else:
            has_function = contains(e, lambda n: isinstance(n, Function))
            bad["unlocated"] += has_function and not contains(out, stays_for_the_mesh_library)
        before, after = evaluate(e, ENV)(()), evaluate(out, ENV)(())
        if not np.allclose(before, after, rtol=1e-4, atol=1e-5):
            bad["value"] += 1
            if bad["value"] <= 3:
                print("value changed, seed", seed, restricted, before, after)
    print(
        f"{trials} integrals over points: {bad['value']} values changed, "
        f"{bad['static']} static types changed, {bad['grammar']} outside N5 or N6, "
        f"{bad['unlocated']} with no pull-back left for the mesh library, "
        f"{bad['unlowered']} of {bad['restricted']} restrictions left unlowered; "
        f"N6 rejected {bad['rejected']} of {bad['lazy']} inputs with a gradient to evaluate"
    )
    return bad


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)


def test_points():
    counts = main(60)
    assert counts["unlocated"] == counts["unlowered"] == 0
    assert counts["restricted"] > 0
    assert_clean(counts)
