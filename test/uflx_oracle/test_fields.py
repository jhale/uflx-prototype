import random
import sys

import numpy as np

from uflx._src.meshes import cell_map
from uflx._src.nodes import Function
from uflx._src.passes import applications, derivatives, integrals
from uflx._src.passes.applications import applications_expanded
from uflx._src.passes.derivatives import _derivatives_complete
from uflx._src.passes.integrals import integrals_expanded
from uflx.arithmetic import add, conjugate, divide, multiply, num
from uflx.calculus import coordinate, directional_derivative, gradient, integral
from uflx.forms import form, function
from uflx.geometry import density, pull_back
from uflx.tensors import index, index_sum, indexed
from uflx_basix.spaces import space
from uflx_oracle.interpreter import Interpreter

from ..testutil import assert_clean, check_types, form_with_fixed_argument
from ..uflx_basix.integrals import GEOMETRY, K
from ..uflx_basix.integrals import D as MESH

VD = space("VD", domain=MESH)
u, w, h = (function(VD, n) for n in "uwh")
BOUND = [function(VD, n) for n in ("v", "y0", "y1", "y2")]
i = index("i")
X = coordinate(MESH)


def field(rnd, names, depth):
    # A scalar field on the mesh, built from the names, u, w and the coordinates.
    if depth == 0 or rnd.random() < 0.35:
        base = rnd.choice([*names, u, w, indexed(X, (rnd.randrange(2),)), num(2)])
        if rnd.random() < 0.3 and isinstance(base, Function):
            return indexed(gradient(base, MESH), (rnd.randrange(2),))
        return base
    a, b = field(rnd, names, depth - 1), field(rnd, names, depth - 1)
    return rnd.choice(
        [
            add(a, b),
            multiply(a, b),
            conjugate(a),
            divide(a, add(num(3), multiply(num(0.2), b))),
            index_sum(
                multiply(indexed(gradient(a, MESH), (i,)), indexed(gradient(b, MESH), (i,))), (i,)
            ),
        ]
    )


def integral_of(rnd, body):
    # The integral over the mesh, or the same written by hand with a pull-back by the cells.
    if rnd.random() < 0.7:
        return integral(body, MESH)
    return sum(
        (
            integral(
                multiply(pull_back(body, cell_map(MESH, slot)), density(cell_map(MESH, slot))), K
            )
            for slot in range(len(MESH.topology.cell_types))
        ),
        num(0),
    )


def random_form(rnd, n, depth):
    names = rnd.sample(BOUND, n)
    body = field(rnd, [], 2)
    for b in names:
        body = multiply(b, body)
    e = form(integral_of(rnd, body), tuple(names)) if names else integral_of(rnd, body)
    if depth > 0 and rnd.random() < 0.5:
        e = directional_derivative(e, u, u, rnd.choice([h, w]))
    if depth > 0 and rnd.random() < 0.6:
        bigger = random_form(rnd, n + 1, depth - 1)
        e = add(e, form_with_fixed_argument(bigger, rnd.randrange(n + 1), field(rnd, [], 1)))
    return e


def main(trials):
    env = GEOMETRY | {u: num(1) + X[0] * X[1], w: X[0] - num(2) * X[1], h: num(0.5) + X[1]}
    args = [num(1) + X[0], X[1] * X[1]]
    bad = dict(value=0, grammar=0)
    for seed in range(trials):
        rnd = random.Random(seed)
        n = rnd.randrange(2)
        e = random_form(rnd, n, 2)
        out = integrals.expand_integrals(
            derivatives.apply_derivatives(applications.expand_applications(e))
        )
        check_types(bad, e, out)
        vals = tuple(args[: len(e.argument_spaces)])
        oracle = Interpreter(env)
        before, after = oracle.evaluate(e, arguments=vals), oracle.evaluate(out, arguments=vals)
        if not np.allclose(before, after, rtol=1e-4, atol=1e-6):
            bad["value"] += 1
            if bad["value"] <= 3:
                print("value changed, seed", seed, before, after)
        bad["grammar"] += not (
            applications_expanded(out) and _derivatives_complete(out) and integrals_expanded(out)
        )
    print(f"{trials} forms on a mesh: {bad['value']} values changed, {bad['grammar']} outside N4")
    return bad


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 60)


def test_fields():
    assert_clean(main(60))
