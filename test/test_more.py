import numpy as np

from uflx_oracle import evaluate
from uflx_spec_interpreter.arithmetic import add, conjugate, divide, multiply
from uflx_spec_interpreter.forms import action, form, function, space
from uflx_spec_interpreter.indices import index
from uflx_spec_interpreter.literals import num, zero
from uflx_spec_interpreter.passes import actions
from uflx_spec_interpreter.static import arity
from uflx_spec_interpreter.tensors import component_tensor, index_sum, indexed, tensor

rng = np.random.default_rng(2)
V, Vd, Q = space("V"), space("V", dual=True), space("Q")
W, Wd = space("W", value_shape=(2,)), space("W", dual=True, value_shape=(2,))
u, v, w, z, h = (function(V, n) for n in "uvwzh")
q, r = function(Q, "q"), function(Q, "r")
U, Y, T, U2 = (function(W, n) for n in ("U", "Y", "T", "U2"))
lW = function(Wd, "lW")
i, j, k = index("i"), index("j"), index("k")
cplx = lambda *s: (
    rng.normal(size=s) + 1j * rng.normal(size=s) if s else complex(*rng.normal(size=2))
)
q2 = function(V, "q2")
env = {f: cplx() for f in (u, v, w, z, h, q, r, q2)}
env |= {f: cplx(2) for f in (U, Y, T, U2)}
cl = cplx(2)
env[lW] = lambda x: np.sum(cl * x)
c, c2 = tensor((num(1), num(2))), tensor((num(3), num(-1)))


def check(name, e, ienv=None, vals=None):
    out = actions.apply_actions(e)
    n = arity(e)
    vals = vals if vals is not None else tuple(cplx() for _ in range(n))
    before = evaluate(e, env, ienv)(vals)
    after = evaluate(out, env, ienv)(vals)
    assert np.allclose(before, after, atol=1e-5), name
    assert actions.in_n2(out), name


def test_cases():
    F = form(multiply(u, conjugate(v)), (conjugate(v), u))
    L1, L2 = form(multiply(v, w), (v,)), form(multiply(num(2), multiply(v, z)), (v,))

    print("-- rows not tested before")
    for kk in range(2):
        check(
            f"zero form, argument with family index i = {kk}",
            action(zero((), (), (V, V)), 1, multiply(indexed(c, (i,)), w)),
            {i: kk},
        )
    check("form divided by a scalar", action(divide(F, num(3)), 0, z))
    check("entry of a tensor of forms", action(indexed(tensor((L1, L2)), (1,)), 0, z))
    check("index sum of forms", action(index_sum(multiply(indexed(c, (i,)), L1), (i,)), 0, z))
    S = form(add(multiply(v, u), multiply(q, w)), ((v, q),))
    check("tuple argument (v, q) := (w, r)", action(S, 0, (w, r)))
    check(
        "tuple argument swapped (v, q2) := (q2, v)",
        action(form(add(multiply(v, u), multiply(q2, w)), ((v, q2),)), 0, (q2, v)),
    )
    B = form(index_sum(multiply(indexed(c, (i,)), v), (i,)), (v,))
    for kk in range(2):
        check(
            f"capture inside the form body, i = {kk}",
            action(B, 0, multiply(indexed(c2, (i,)), w)),
            {i: kk},
        )

    print("-- vector-valued spaces")
    M = form(
        index_sum(multiply(indexed(U, (i,)), conjugate(indexed(T, (i,)))), (i,)), (conjugate(T), U)
    )
    check("vector mass form, actions at 1 and 0", action(action(M, 1, tensor((w, z))), 0, U2))
    check(
        "vector form, argument a component tensor",
        action(M, 1, component_tensor(multiply(indexed(c, (k,)), w), (k,))),
        vals=(cplx(2),),
    )
    check("vector pairing remains", action(form(multiply(action(lW, 0, Y), u), (Y,)), 0, U2))
    G = form(index_sum(multiply(indexed(Y, (j,)), indexed(U, (j,))), (j,)), (Y,))
    for kk in range(2):
        check(
            f"vector argument with family index j, body binds j, j = {kk}",
            action(G, 0, component_tensor(multiply(indexed(c, (j,)), indexed(U2, (k,))), (k,))),
            {j: kk},
        )
