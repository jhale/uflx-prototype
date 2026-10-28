import numpy as np

from uflx_spec_interpreter.nodes import (
    Action,
    ComponentTensor,
    Conjugate,
    Division,
    Form,
    Function,
    Index,
    Indexed,
    IndexSum,
    Num,
    Product,
    Space,
    Sum,
    Tensor,
    Zero,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import actions
from uflx_spec_interpreter.static import arity

rng = np.random.default_rng(2)
V, Vd, Q = Space("V"), Space("V", dual=True), Space("Q")
W, Wd = Space("W", value_shape=(2,)), Space("W", dual=True, value_shape=(2,))
u, v, w, z, h = (Function(V, n) for n in "uvwzh")
q, r = Function(Q, "q"), Function(Q, "r")
U, Y, T, U2 = (Function(W, n) for n in ("U", "Y", "T", "U2"))
lW = Function(Wd, "lW")
i, j, k = Index("i"), Index("j"), Index("k")
cplx = lambda *s: (
    rng.normal(size=s) + 1j * rng.normal(size=s) if s else complex(*rng.normal(size=2))
)
q2 = Function(V, "q2")
env = {f: cplx() for f in (u, v, w, z, h, q, r, q2)}
env |= {f: cplx(2) for f in (U, Y, T, U2)}
cl = cplx(2)
env[lW] = lambda x: np.sum(cl * x)
c, c2 = Tensor((Num(1), Num(2))), Tensor((Num(3), Num(-1)))


def check(name, e, ienv=None, vals=None):
    out = actions.apply_actions(e)
    n = arity(e)
    vals = vals if vals is not None else tuple(cplx() for _ in range(n))
    before = evaluate(e, env, ienv)(vals)
    after = evaluate(out, env, ienv)(vals)
    assert np.allclose(before, after, atol=1e-5), name
    assert actions.in_n2(out), name


def test_cases():
    F = Form(Product(u, Conjugate(v)), (Conjugate(v), u))
    L1, L2 = Form(Product(v, w), (v,)), Form(Product(Num(2), Product(v, z)), (v,))

    print("-- rows not tested before")
    for kk in range(2):
        check(
            f"zero form, argument with family index i = {kk}",
            Action(Zero((), (), (V, V)), 1, Product(Indexed(c, (i,)), w)),
            {i: kk},
        )
    check("form divided by a scalar", Action(Division(F, Num(3)), 0, z))
    check("entry of a tensor of forms", Action(Indexed(Tensor((L1, L2)), (1,)), 0, z))
    check("index sum of forms", Action(IndexSum(Product(Indexed(c, (i,)), L1), (i,)), 0, z))
    S = Form(Sum(Product(v, u), Product(q, w)), ((v, q),))
    check("tuple argument (v, q) := (w, r)", Action(S, 0, (w, r)))
    check(
        "tuple argument swapped (v, q2) := (q2, v)",
        Action(Form(Sum(Product(v, u), Product(q2, w)), ((v, q2),)), 0, (q2, v)),
    )
    B = Form(IndexSum(Product(Indexed(c, (i,)), v), (i,)), (v,))
    for kk in range(2):
        check(
            f"capture inside the form body, i = {kk}",
            Action(B, 0, Product(Indexed(c2, (i,)), w)),
            {i: kk},
        )

    print("-- vector-valued spaces")
    M = Form(
        IndexSum(Product(Indexed(U, (i,)), Conjugate(Indexed(T, (i,)))), (i,)), (Conjugate(T), U)
    )
    check("vector mass form, actions at 1 and 0", Action(Action(M, 1, Tensor((w, z))), 0, U2))
    check(
        "vector form, argument a component tensor",
        Action(M, 1, ComponentTensor(Product(Indexed(c, (k,)), w), (k,))),
        vals=(cplx(2),),
    )
    check("vector pairing remains", Action(Form(Product(Action(lW, 0, Y), u), (Y,)), 0, U2))
    G = Form(IndexSum(Product(Indexed(Y, (j,)), Indexed(U, (j,))), (j,)), (Y,))
    for kk in range(2):
        check(
            f"vector argument with family index j, body binds j, j = {kk}",
            Action(G, 0, ComponentTensor(Product(Indexed(c, (j,)), Indexed(U2, (k,))), (k,))),
            {j: kk},
        )
