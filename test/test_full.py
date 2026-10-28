import itertools
import random
import sys

import numpy as np
from testutil import assert_clean, contains

from uflx_spec_interpreter.nodes import (
    CT,
    Abs,
    Action,
    Conj,
    D,
    Div,
    Fn,
    Form,
    Im,
    Index,
    Indexed,
    Inv,
    ISum,
    Log,
    Num,
    Power,
    Prod,
    Re,
    Space,
    Sqrt,
    Sum,
    Tensor,
    Zero,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import actions, derivatives
from uflx_spec_interpreter.static import arity, free, shape

V, Vd = Space("V"), Space("V", dual=True)
W, Wd = Space("W", shape=(2,)), Space("W", dual=True, shape=(2,))
u, v, w, z, h, q = (Fn(V, n) for n in "uvwzhq")
U, U2, U3 = (Fn(W, n) for n in ("U", "U2", "U3"))
lV, lW = Fn(Vd, "lV"), Fn(Wd, "lW")
BOUND = {
    "V": [u, v, q] + [Fn(V, f"y{k}") for k in range(4)],
    "W": [U2] + [Fn(W, f"Y{k}") for k in range(3)],
}
kf, m = Index("kf"), Index("m")
c = Tensor((Num(1), Num(2)))


def dot(X, Y):
    return ISum(Prod(Indexed(X, (m,)), Indexed(Y, (m,))), (m,))


def scalar(rnd, depth, nd=0):
    if depth == 0 or rnd.random() < 0.3:
        pick = rnd.random()
        if pick < 0.1:
            return Num(rnd.choice([1, 2, -3, 0.5]))
        if pick < 0.2:
            return Indexed(rnd.choice([U, U2, U3]), (rnd.randrange(2),))
        if pick < 0.27:
            return Action(lV, 0, rnd.choice([u, w, z]))
        if pick < 0.32:
            return Action(lW, 0, rnd.choice([U, U2]))
        return rnd.choice([u, v, w, z, h, q])
    a, b = scalar(rnd, depth - 1, nd), scalar(rnd, depth - 1, nd)
    near = lambda x: Sum(Num(3), Prod(Num(0.2), x))
    ops = [
        lambda: Sum(a, b),
        lambda: Prod(a, b),
        lambda: Conj(a),
        lambda: Div(a, near(b)),
        lambda: Re(a),
        lambda: Im(a),
        lambda: Abs(near(a)),
        lambda: Sqrt(near(a)),
        lambda: Log(near(a)),
        lambda: Power(near(a), rnd.choice([Num(2), Num(0.5), Prod(Num(0.3), b)])),
        # A binder of kf, the index that the families of arguments and directions use.
        lambda: ISum(Prod(Indexed(c, (kf,)), a), (kf,)),
        lambda: Indexed(
            Inv(
                Tensor((Tensor((near(a), Prod(Num(0.1), b))), Tensor((Prod(Num(0.1), a), near(b)))))
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
        return Tensor((scalar(rnd, depth), scalar(rnd, depth)))
    if pick == 1:
        return rnd.choice([U, U2, U3])
    if pick == 2:
        return CT(Prod(Indexed(c, (kf,)), Indexed(U3, (m,))), (m,))
    return CT(Prod(scalar(rnd, depth), Indexed(U, (m,))), (m,))


def wrap(rnd, xs, body):
    # Put the variable inside each kind of node, so every row of the derivative pass is reached.
    near = lambda y: Sum(Num(3), Prod(Num(0.2), y))
    xb = Prod(xs, body)
    return rnd.choice(
        [
            xb,
            Sqrt(near(xb)),
            Log(near(xb)),
            Abs(near(xb)),
            Power(near(xs), Prod(Num(0.3), Sum(xs, body))),
            Power(near(xb), Num(2)),
            Div(body, near(xs)),
            Re(xb),
            Indexed(
                Inv(
                    Tensor(
                        (
                            Tensor((near(xs), Prod(Num(0.1), body))),
                            Tensor((Prod(Num(0.1), xs), near(body))),
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
        f = wrap(rnd, x, body) if arity(body) == 0 else Prod(x, body)
        point = rnd.choice([x, x, z, Sum(x, Num(0.1))])
        direction = rnd.choice([h, z, Num(1), Prod(Indexed(c, (kf,)), h)])
    else:
        x = U2
        f = wrap(rnd, dot(U2, U), body) if arity(body) == 0 else Prod(dot(U2, U), body)
        point = rnd.choice([U2, U2, U3, Tensor((z, h))])
        direction = rnd.choice(
            [Tensor((h, z)), CT(Prod(Indexed(c, (kf,)), Indexed(U3, (m,))), (m,))]
        )
    return D(f, x, point, direction)


def argument(rnd, spec):
    if isinstance(spec, tuple):
        return tuple(argument(rnd, s) for s in spec)
    return scalar(rnd, 2) if spec == "V" else vector(rnd, 1)


def entry_value(rng, spec):
    if isinstance(spec, tuple):
        return tuple(entry_value(rng, s) for s in spec)
    if spec == "V":
        return 0.3 * complex(*rng.normal(size=2))
    return 0.3 * (rng.normal(size=2) + 1j * rng.normal(size=2))


def leaf_form(rnd, specs):
    used, entries, body = set(), [], scalar(rnd, 2)
    for spec in specs:
        members = []
        for s in spec if isinstance(spec, tuple) else (spec,):
            fn = rnd.choice([b for b in BOUND[s] if b not in used])
            used.add(fn)
            body = Prod(fn if s == "V" else dot(fn, U), body)
            members.append(Conj(fn) if rnd.random() < 0.3 else fn)
        entries.append(tuple(members) if isinstance(spec, tuple) else members[0])
    return Form(body, tuple(entries))


def form(rnd, specs, depth):
    if depth == 0 or rnd.random() < 0.25:
        return leaf_form(rnd, specs) if specs else scalar(rnd, 2)
    op = rnd.randrange(9)
    if op == 8:
        return Zero((), (), tuple(Space("arg") for _ in specs))
    if op == 0:
        return Sum(form(rnd, specs, depth - 1), form(rnd, specs, depth - 1))
    if op == 1:
        k = rnd.randrange(len(specs) + 1)
        return Prod(form(rnd, specs[:k], depth - 1), form(rnd, specs[k:], depth - 1))
    if op == 2:
        return Conj(form(rnd, specs, depth - 1))
    if op == 3:
        return Div(form(rnd, specs, depth - 1), Num(3))
    if op == 4:
        return derivative(rnd, form(rnd, specs, depth - 1), 1)
    if op == 5:
        return ISum(Prod(Indexed(c, (kf,)), form(rnd, specs, depth - 1)), (kf,))
    if op == 6:
        return Indexed(
            Tensor((form(rnd, specs, depth - 1), form(rnd, specs, depth - 1))), (rnd.randrange(2),)
        )
    spec = rnd.choice(["V", "W", ("V", "W"), ("V", "V")])
    p = rnd.randrange(len(specs) + 1)
    inner = [*specs[:p], spec, *specs[p:]]
    return Action(form(rnd, inner, depth - 1), p, argument(rnd, spec))


def main(trials):
    rng = np.random.default_rng(0)
    env = {f: 0.3 * complex(*rng.normal(size=2)) for f in [u, v, w, z, h, q] + BOUND["V"]}
    env |= {
        f: 0.3 * (rng.normal(size=2) + 1j * rng.normal(size=2)) for f in [U, U2, U3] + BOUND["W"]
    }
    cv, cw = complex(*rng.normal(size=2)), rng.normal(size=2) + 1j * rng.normal(size=2)
    env[lV] = lambda x: cv * np.conj(x)
    env[lW] = lambda x: np.sum(cw * np.conj(x))
    bad = dict(value=0, static=0, grammar=0)
    lazy, rejected = 0, 0
    for seed in range(trials):
        rnd = random.Random(seed)
        specs = rnd.choice([[], ["V"], ["W"], [("V", "W")], ["V", "W"]])
        e = form(rnd, specs, 3)
        out = derivatives.apply_derivatives(actions.apply_actions(e))
        if (shape(e), free(e), arity(e)) != (shape(out), free(out), arity(out)):
            bad["static"] += 1
            if bad["static"] <= 2:
                print(
                    "static type changed, seed", seed, (shape(e), free(e)), (shape(out), free(out))
                )
        if contains(
            e,
            lambda n: (
                isinstance(n, D)
                or (
                    lambda n: isinstance(n, Action) and not (isinstance(n.f, Fn) and n.f.space.dual)
                )(n)
            ),
        ):
            lazy += 1
            rejected += not (actions.in_n2(e) and derivatives.in_n3(e))
        if not (actions.in_n2(out) and derivatives.in_n3(out)):
            bad["grammar"] += 1
        vals = tuple(entry_value(rng, s) for s in specs)
        fr = free(e)
        for combo in itertools.product(*(range(n) for n in fr.values())):
            ienv = dict(zip(fr, combo))
            before, after = evaluate(e, env, ienv)(vals), evaluate(out, env, ienv)(vals)
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
