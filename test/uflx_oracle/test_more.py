import numpy as np

from uflx._src.literals import zero
from uflx._src.passes import applications
from uflx._src.passes.applications import applications_expanded
from uflx.arithmetic import add, conjugate, divide, multiply, num
from uflx.forms import apply, dual_space, form, function, partial
from uflx.nodes import ScalarType
from uflx.tensors import component_tensor, index, index_sum, indexed, tensor
from uflx_basix.spaces import space
from uflx_oracle.interpreter import Interpreter

from ..testutil import ill_typed

rng = np.random.default_rng(2)
V, Vd, Q = space("V"), dual_space(space("V")), space("Q")
W, Wd = space("W", value_shape=(2,)), dual_space(space("W", value_shape=(2,)))
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
    out = applications.expand_applications(e)
    n = len(e.argument_spaces)
    vals = vals if vals is not None else tuple(cplx() for _ in range(n))
    oracle = Interpreter(env)
    before = oracle.evaluate(e, arguments=vals, indices=ienv)
    after = oracle.evaluate(out, arguments=vals, indices=ienv)
    assert np.allclose(before, after, atol=1e-5), name
    assert applications_expanded(out), name
    assert not (ill_typed(e) or ill_typed(out)), (name, ill_typed(e) or ill_typed(out))


def test_cases():
    F = form(multiply(u, conjugate(v)), (conjugate(v), u))
    L1, L2 = form(multiply(v, w), (v,)), form(multiply(num(2), multiply(v, z)), (v,))

    print("-- rows not tested before")
    for kk in range(2):
        check(
            f"zero form, argument with family index i = {kk}",
            form(
                apply(zero((), (), (V, V), ScalarType.real), (v, multiply(indexed(c, (i,)), w))),
                (v,),
            ),
            {i: kk},
        )
    check("form divided by a scalar", form(apply(divide(F, num(3)), (z, u)), (u,)))
    check("entry of a tensor of forms", apply(indexed(tensor((L1, L2)), (1,)), (z,)))
    check("index sum of forms", apply(index_sum(multiply(indexed(c, (i,)), L1), (i,)), (z,)))
    S = form(add(multiply(v, u), multiply(q, w)), ((v, q),))
    check("tuple argument (v, q) := (w, r)", apply(S, ((w, r),)))
    check(
        "tuple argument swapped (v, q2) := (q2, v)",
        apply(form(add(multiply(v, u), multiply(q2, w)), ((v, q2),)), ((q2, v),)),
    )
    B = form(index_sum(multiply(indexed(c, (i,)), v), (i,)), (v,))
    for kk in range(2):
        check(
            f"capture inside the form body, i = {kk}",
            apply(B, (multiply(indexed(c2, (i,)), w),)),
            {i: kk},
        )

    print("-- vector-valued spaces")
    M = form(
        index_sum(multiply(indexed(U, (i,)), conjugate(indexed(T, (i,)))), (i,)), (conjugate(T), U)
    )
    check("vector mass form, actions at 1 and 0", apply(M, (U2, tensor((w, z)))))
    check(
        "vector form, argument a component tensor",
        partial(M, {U: component_tensor(multiply(indexed(c, (k,)), w), (k,))}),
        vals=(cplx(2),),
    )
    check("vector pairing remains", apply(form(multiply(apply(lW, (Y,)), u), (Y,)), (U2,)))
    G = form(index_sum(multiply(indexed(Y, (j,)), indexed(U, (j,))), (j,)), (Y,))
    for kk in range(2):
        check(
            f"vector argument with family index j, body binds j, j = {kk}",
            apply(G, (component_tensor(multiply(indexed(c, (j,)), indexed(U2, (k,))), (k,)),)),
            {j: kk},
        )
