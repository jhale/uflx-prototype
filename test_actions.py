import numpy as np

import passes
from core import CT, D, Action, Conj, Fn, Form, Index, Indexed, Num, Prod, Space, Sum, Tensor
from oracle import evaluate

rng = np.random.default_rng(1)
V, Vd = Space("V"), Space("V", dual=True)
u, v, w, z, h = (Fn(V, n) for n in "uvwzh")
l = Fn(Vd, "l")
i, j = Index("i"), Index("j")
env = {f: complex(*rng.normal(size=2)) for f in (u, v, w, z, h)}
cl = complex(*rng.normal(size=2))
env[l] = lambda x: cl * np.conj(x)
args = tuple(complex(*rng.normal(size=2)) for _ in range(3))


def check(name, e, ienv=None):
    ienv = ienv or {}
    out = passes.apply_actions(e)
    from static import arity

    n = arity(e)
    before = evaluate(e, env, ienv)(args[:n])
    after = evaluate(out, env, ienv)(args[:n])
    ok = np.allclose(before, after, atol=1e-6)
    print(f"{name:45s} value kept: {ok}   in N2: {passes.in_n2(out)}")
    return out


a = Prod(u, Conj(v))
F = Form(a, (Conj(v), u))
check("bilinear form, action at 1 then 0", Action(Action(F, 1, w), 0, z))
check("action at 0 leaves a linear form", Action(F, 0, z))
check("sum and product of forms", Action(Sum(Prod(Num(3), F), F), 1, w))
check("product of two linear forms", Action(Prod(Form(Prod(v, Num(2)), (v,)), Form(Conj(u), (Conj(u),))), 1, w))
check("conjugate of a form", Action(Conj(F), 0, z))
check("pairing remains", Action(Form(Prod(Action(l, 0, v), u), (v, u)), 0, w))

# The family index i of g must not be captured by the ComponentTensor that binds i.
c = Tensor((Num(1), Num(2)))
g = Prod(Indexed(c, (i,)), w)
G = CT(Prod(Indexed(Tensor((Num(5), Num(7))), (i,)), Form(Prod(v, Num(1)), (v,))), (i,))
for k in range(2):
    check(f"capture, family index i = {k}", Action(G, 0, g), {i: k})

# The argument contains the variable of the derivative: expect h * u, not 2 h u.
J = Form(D(Prod(u, v), u, u, h), (v,))
out = check("action on a form with a derivative of u at u", Action(J, 0, u))
print("   value:", evaluate(out, env)(()), " h * u:", env[h] * env[u])
