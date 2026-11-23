import itertools
import random
import sys

import numpy as np

from uflx._src.literals import zero
from uflx._src.nodes import Apply, DirectionalDerivative
from uflx._src.passes import applications, derivatives
from uflx._src.passes.applications import applications_expanded
from uflx._src.passes.derivatives import _derivatives_complete
from uflx.arithmetic import (
    absolute,
    add,
    conjugate,
    divide,
    imaginary_part,
    logarithm,
    multiply,
    num,
    power,
    real_part,
    square_root,
)
from uflx.calculus import directional_derivative
from uflx.forms import apply, conjugate_space, dual_space, form, function
from uflx.nodes import ScalarType
from uflx.tensors import component_tensor, index, index_sum, indexed, inverse, tensor
from uflx_jax_compiler import trace

from ..testutil import (
    assert_clean,
    check_types,
    contains,
    form_with_fixed_argument,
    keeps_static_type,
    pairing,
)
from ..uflx.spaces import space

V, Vd = space("V"), dual_space(space("V"))
W, Wd = space("W", value_shape=(2,)), dual_space(space("W", value_shape=(2,)))
u, v, w, z, h, q = (function(V, n) for n in "uvwzhq")
U, U2, U3 = (function(W, n) for n in ("U", "U2", "U3"))
lV, lW = function(Vd, "lV"), function(Wd, "lW")


def bound(base):
    # The functions that a random form binds, for the base name of a space.
    if base == "V":
        return [u, v, q] + [function(V, f"y{k}") for k in range(4)]
    return [U2] + [function(W, f"Y{k}") for k in range(3)]


kf, m = index("kf"), index("m")
c = tensor((num(1), num(2)))


def dot(X, Y):
    return index_sum(multiply(indexed(X, (m,)), indexed(Y, (m,))), (m,))


def scalar(rnd, depth, nd=0):
    if depth == 0 or rnd.random() < 0.3:
        pick = rnd.random()
        if pick < 0.1:
            return num(rnd.choice([1, 2, -3, 0.5]))
        if pick < 0.2:
            return indexed(rnd.choice([U, U2, U3]), (rnd.randrange(2),))
        if pick < 0.27:
            return apply(lV, (rnd.choice([u, w, z]),))
        if pick < 0.32:
            return apply(lW, (rnd.choice([U, U2]),))
        return rnd.choice([u, v, w, z, h, q])
    a, b = scalar(rnd, depth - 1, nd), scalar(rnd, depth - 1, nd)
    near = lambda x: add(num(3), multiply(num(0.2), x))
    ops = [
        lambda: add(a, b),
        lambda: multiply(a, b),
        lambda: conjugate(a),
        lambda: divide(a, near(b)),
        lambda: real_part(a),
        lambda: imaginary_part(a),
        lambda: absolute(near(a)),
        lambda: square_root(near(a)),
        lambda: logarithm(near(a)),
        lambda: power(near(a), rnd.choice([num(2), num(0.5), multiply(num(0.3), b)])),
        # A binder of kf, the index that the families of arguments and directions use.
        lambda: index_sum(multiply(indexed(c, (kf,)), a), (kf,)),
        lambda: indexed(
            inverse(
                tensor(
                    (
                        tensor((near(a), multiply(num(0.1), b))),
                        tensor((multiply(num(0.1), a), near(b))),
                    )
                )
            ),
            (rnd.randrange(2), rnd.randrange(2)),
        ),
    ]
    if nd < 2:
        ops.append(lambda: derivative(rnd, scalar(rnd, depth - 1, nd + 1), nd + 1))
    return rnd.choice(ops)()


def vector(rnd, depth):
    pick = rnd.randrange(4)
    if pick == 0:
        return tensor((scalar(rnd, depth), scalar(rnd, depth)))
    if pick == 1:
        return rnd.choice([U, U2, U3])
    if pick == 2:
        return component_tensor(multiply(indexed(c, (kf,)), indexed(U3, (m,))), (m,))
    return component_tensor(multiply(scalar(rnd, depth), indexed(U, (m,))), (m,))


def wrap(rnd, xs, body):
    # Put the variable inside each kind of node, so every row of the derivative pass is reached.
    near = lambda y: add(num(3), multiply(num(0.2), y))
    xb = multiply(xs, body)
    return rnd.choice(
        [
            xb,
            square_root(near(xb)),
            logarithm(near(xb)),
            absolute(near(xb)),
            power(near(xs), multiply(num(0.3), add(xs, body))),
            power(near(xb), num(2)),
            divide(body, near(xs)),
            real_part(xb),
            indexed(
                inverse(
                    tensor(
                        (
                            tensor((near(xs), multiply(num(0.1), body))),
                            tensor((multiply(num(0.1), xs), near(body))),
                        )
                    )
                ),
                (rnd.randrange(2), rnd.randrange(2)),
            ),
        ]
    )


def derivative(rnd, body, nd):
    if rnd.random() < 0.7:
        x = rnd.choice([u, v, w])
        f = wrap(rnd, x, body) if len(body.argument_spaces) == 0 else multiply(x, body)
        point = rnd.choice([x, x, z, add(x, num(0.1))])
        direction = rnd.choice([h, z, num(1), multiply(indexed(c, (kf,)), h)])
    else:
        x = U2
        f = (
            wrap(rnd, dot(U2, U), body)
            if len(body.argument_spaces) == 0
            else multiply(dot(U2, U), body)
        )
        point = rnd.choice([U2, U2, U3, tensor((z, h))])
        direction = rnd.choice(
            [tensor((h, z)), component_tensor(multiply(indexed(c, (kf,)), indexed(U3, (m,))), (m,))]
        )
    return directional_derivative(f, x, point, direction)


# A spec names the space of an argument, "V" or "W", with a star for an entry conjugate(v).
def base(spec):
    return spec.rstrip("*")


def flip(spec):
    # The spec of the conjugate of a form, whose linear arguments become antilinear.
    if isinstance(spec, tuple):
        return tuple(map(flip, spec))
    return base(spec) if spec.endswith("*") else spec + "*"


def argument(rnd, spec):
    if isinstance(spec, tuple):
        return tuple(argument(rnd, s) for s in spec)
    return scalar(rnd, 2) if base(spec) == "V" else vector(rnd, 1)


def entry_value(rng, spec):
    if isinstance(spec, tuple):
        return tuple(entry_value(rng, s) for s in spec)
    if base(spec) == "V":
        return 0.3 * complex(*rng.normal(size=2))
    return 0.3 * (rng.normal(size=2) + 1j * rng.normal(size=2))


def leaf_form(rnd, specs):
    used, entries, body = set(), [], scalar(rnd, 2)
    for spec in specs:
        members = []
        for s in spec if isinstance(spec, tuple) else (spec,):
            fn = rnd.choice([b for b in bound(base(s)) if b not in used])
            used.add(fn)
            entry = conjugate(fn) if s.endswith("*") else fn
            body = multiply(entry if base(s) == "V" else dot(entry, U), body)
            members.append(entry)
        entries.append(tuple(members) if isinstance(spec, tuple) else members[0])
    return form(body, tuple(entries))


def spaces(spec):
    if isinstance(spec, tuple):
        return tuple(map(spaces, spec))
    space = V if base(spec) == "V" else W
    return conjugate_space(space) if spec.endswith("*") else space


def random_form(rnd, specs, depth):
    if depth == 0 or rnd.random() < 0.25:
        return leaf_form(rnd, specs) if specs else scalar(rnd, 2)
    op = rnd.randrange(9)
    if op == 8:
        return zero((), (), tuple(spaces(spec) for spec in specs), ScalarType.real)
    if op == 0:
        return add(random_form(rnd, specs, depth - 1), random_form(rnd, specs, depth - 1))
    if op == 1:
        k = rnd.randrange(len(specs) + 1)
        return multiply(
            random_form(rnd, specs[:k], depth - 1), random_form(rnd, specs[k:], depth - 1)
        )
    if op == 2:
        return conjugate(random_form(rnd, [flip(spec) for spec in specs], depth - 1))
    if op == 3:
        return divide(random_form(rnd, specs, depth - 1), num(3))
    if op == 4:
        return derivative(rnd, random_form(rnd, specs, depth - 1), 1)
    if op == 5:
        return index_sum(multiply(indexed(c, (kf,)), random_form(rnd, specs, depth - 1)), (kf,))
    if op == 6:
        return indexed(
            tensor((random_form(rnd, specs, depth - 1), random_form(rnd, specs, depth - 1))),
            (rnd.randrange(2),),
        )
    spec = rnd.choice(["V", "W", "V*", ("V", "W"), ("V", "V"), ("V*", "W")])
    p = rnd.randrange(len(specs) + 1)
    inner = [*specs[:p], spec, *specs[p:]]
    return form_with_fixed_argument(random_form(rnd, inner, depth - 1), p, argument(rnd, spec))


def main(trials):
    rng = np.random.default_rng(0)
    env = {f: 0.3 * complex(*rng.normal(size=2)) for f in [u, v, w, z, h, q, *bound("V")]}
    env |= {
        f: 0.3 * (rng.normal(size=2) + 1j * rng.normal(size=2)) for f in [U, U2, U3, *bound("W")]
    }
    cv, cw = complex(*rng.normal(size=2)), rng.normal(size=2) + 1j * rng.normal(size=2)
    env[lV] = lambda x: cv * x
    env[lW] = lambda x: np.sum(cw * x)
    bad = dict(value=0, static=0, grammar=0)
    lazy, rejected = 0, 0
    for seed in range(trials):
        rnd = random.Random(seed)
        specs = rnd.choice([[], ["V"], ["W"], ["V*"], [("V", "W")], ["V", "W"], ["V*", "W*"]])
        e = random_form(rnd, specs, 3)
        out = derivatives.apply_derivatives(applications.expand_applications(e))
        check_types(bad, e, out)
        if not keeps_static_type(e, out):
            bad["static"] += 1
            if bad["static"] <= 2:
                print(
                    "static type changed, seed",
                    seed,
                    (e.value_shape, dict(e.free_indices)),
                    (out.value_shape, dict(out.free_indices)),
                )
        if contains(
            e,
            lambda n: (
                isinstance(n, DirectionalDerivative)
                or (lambda n: isinstance(n, Apply) and not pairing(n))(n)
            ),
        ):
            lazy += 1
            rejected += not (applications_expanded(e) and _derivatives_complete(e))
        if not (applications_expanded(out) and _derivatives_complete(out)):
            bad["grammar"] += 1
        vals = tuple(entry_value(rng, s) for s in specs)
        fr = dict(e.free_indices)
        for combo in itertools.product(*(range(n) for n in fr.values())):
            ienv = dict(zip(fr, combo))
            before, after = (
                trace(e, *env, *ienv)(*env.values(), *ienv.values(), *vals),
                trace(out, *env, *ienv)(*env.values(), *ienv.values(), *vals),
            )
            if not np.allclose(before, after, rtol=1e-3, atol=1e-4):
                bad["value"] += 1
                if bad["value"] <= 3:
                    print("value changed, seed", seed, ienv, before, after)
                break
    print(
        f"{trials} expressions: {bad['value']} values changed, "
        f"{bad['static']} static types changed, "
        f"{bad['grammar']} outside N3; "
        f"N3 rejected {rejected} of {lazy} inputs with an action or a derivative"
    )
    return bad | dict(lazy=lazy, rejected=rejected)


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 500)


def test_full():
    assert_clean(main(500))
