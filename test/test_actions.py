import numpy as np

from uflx_spec_interpreter.nodes import (
    Action,
    ComponentTensor,
    Conjugate,
    DirectionalDerivative,
    Form,
    Function,
    Index,
    Indexed,
    Num,
    Product,
    Space,
    Sum,
    Tensor,
)
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import actions
from uflx_spec_interpreter.static import arity

rng = np.random.default_rng(1)
V, Vd = Space("V"), Space("V", dual=True)
u, v, w, z, h = (Function(V, n) for n in "uvwzh")
l = Function(Vd, "l")
i, j = Index("i"), Index("j")
env = {f: complex(*rng.normal(size=2)) for f in (u, v, w, z, h)}
cl = complex(*rng.normal(size=2))
env[l] = lambda x: cl * x
args = tuple(complex(*rng.normal(size=2)) for _ in range(3))


def check(name, e, ienv=None):
    ienv = ienv or {}
    out = actions.apply_actions(e)
    n = arity(e)
    before = evaluate(e, env, ienv)(args[:n])
    after = evaluate(out, env, ienv)(args[:n])
    assert np.allclose(before, after, atol=1e-6), name
    assert actions.in_n2(out), name
    return out


def test_cases():
    a = Product(u, Conjugate(v))
    F = Form(a, (Conjugate(v), u))
    check("bilinear form, action at 1 then 0", Action(Action(F, 1, w), 0, z))
    check("action at 0 leaves a linear form", Action(F, 0, z))
    check("sum and product of forms", Action(Sum(Product(Num(3), F), F), 1, w))
    check(
        "product of two linear forms",
        Action(Product(Form(Product(v, Num(2)), (v,)), Form(Conjugate(u), (Conjugate(u),))), 1, w),
    )
    check("conjugate of a form", Action(Conjugate(F), 0, z))
    check("pairing remains", Action(Form(Product(Action(l, 0, v), u), (v, u)), 0, w))

    # The family index i of g must not be captured by the ComponentTensor that binds i.
    c = Tensor((Num(1), Num(2)))
    g = Product(Indexed(c, (i,)), w)
    G = ComponentTensor(
        Product(Indexed(Tensor((Num(5), Num(7))), (i,)), Form(Product(v, Num(1)), (v,))), (i,)
    )
    for k in range(2):
        check(f"capture, family index i = {k}", Action(G, 0, g), {i: k})

    # The argument contains the variable of the derivative: expect h * u, not 2 h u.
    J = Form(DirectionalDerivative(Product(u, v), u, u, h), (v,))
    out = check("action on a form with a derivative of u at u", Action(J, 0, u))
    assert np.isclose(evaluate(out, env)(()), env[h] * env[u])
