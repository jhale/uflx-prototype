import numpy as np

import passes
from core import CT, Action, Conj, Div, Fn, Form, Index, Indexed, ISum, Num, Prod, Space, Sum, Tensor, Zero
from static import arity
from oracle import evaluate

rng = np.random.default_rng(2)
V, Vd, Q = Space("V"), Space("V", dual=True), Space("Q")
W, Wd = Space("W", shape=(2,)), Space("W", dual=True, shape=(2,))
u, v, w, z, h = (Fn(V, n) for n in "uvwzh")
q, r = Fn(Q, "q"), Fn(Q, "r")
U, Y, T, U2 = (Fn(W, n) for n in ("U", "Y", "T", "U2"))
lW = Fn(Wd, "lW")
i, j, k = Index("i"), Index("j"), Index("k")
cplx = lambda *s: rng.normal(size=s) + 1j * rng.normal(size=s) if s else complex(*rng.normal(size=2))
q2 = Fn(V, "q2")
env = {f: cplx() for f in (u, v, w, z, h, q, r, q2)}
env |= {f: cplx(2) for f in (U, Y, T, U2)}
cl = cplx(2)
env[lW] = lambda x: np.sum(cl * np.conj(x))
c, c2 = Tensor((Num(1), Num(2))), Tensor((Num(3), Num(-1)))


def check(name, e, ienv=None, vals=None):
    out = passes.apply_actions(e)
    n = arity(e)
    vals = vals if vals is not None else tuple(cplx() for _ in range(n))
    before = evaluate(e, env, ienv)(vals)
    after = evaluate(out, env, ienv)(vals)
    ok = np.allclose(before, after, atol=1e-5)
    print(f"{name:55s} value kept: {ok}   in N2: {passes.in_n2(out)}")


F = Form(Prod(u, Conj(v)), (Conj(v), u))
L1, L2 = Form(Prod(v, w), (v,)), Form(Prod(Num(2), Prod(v, z)), (v,))

print("-- rows not tested before")
for kk in range(2):
    check(f"zero form, argument with family index i = {kk}", Action(Zero((), (), (V, V)), 1, Prod(Indexed(c, (i,)), w)), {i: kk})
check("form divided by a scalar", Action(Div(F, Num(3)), 0, z))
check("entry of a tensor of forms", Action(Indexed(Tensor((L1, L2)), (1,)), 0, z))
check("index sum of forms", Action(ISum(Prod(Indexed(c, (i,)), L1), (i,)), 0, z))
S = Form(Sum(Prod(v, u), Prod(q, w)), ((v, q),))
check("tuple argument (v, q) := (w, r)", Action(S, 0, (w, r)))
check("tuple argument swapped (v, q2) := (q2, v)", Action(Form(Sum(Prod(v, u), Prod(q2, w)), ((v, q2),)), 0, (q2, v)))
B = Form(ISum(Prod(Indexed(c, (i,)), v), (i,)), (v,))
for kk in range(2):
    check(f"capture inside the form body, i = {kk}", Action(B, 0, Prod(Indexed(c2, (i,)), w)), {i: kk})

print("-- vector-valued spaces")
M = Form(ISum(Prod(Indexed(U, (i,)), Conj(Indexed(T, (i,)))), (i,)), (Conj(T), U))
check("vector mass form, actions at 1 and 0", Action(Action(M, 1, Tensor((w, z))), 0, U2))
check("vector form, argument a component tensor", Action(M, 1, CT(Prod(Indexed(c, (k,)), w), (k,))), vals=(cplx(2),))
check("vector pairing remains", Action(Form(Prod(Action(lW, 0, Y), u), (Y,)), 0, U2))
G = Form(ISum(Prod(Indexed(Y, (j,)), Indexed(U, (j,))), (j,)), (Y,))
for kk in range(2):
    check(f"vector argument with family index j, body binds j, j = {kk}",
          Action(G, 0, CT(Prod(Indexed(c, (j,)), Indexed(U2, (k,))), (k,))), {j: kk})
