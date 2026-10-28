import numpy as np

from uflx_spec_interpreter.algebra import (
    add,
    component_tensor,
    conjugate,
    index,
    indexed,
    multiply,
    num,
    tensor,
)
from uflx_spec_interpreter.calculus import directional_derivative
from uflx_spec_interpreter.forms import action, form, function, space
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import actions
from uflx_spec_interpreter.static import arity

rng = np.random.default_rng(1)
V, Vd = space("V"), space("V", dual=True)
u, v, w, z, h = (function(V, n) for n in "uvwzh")
l = function(Vd, "l")
i, j = index("i"), index("j")
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
    a = multiply(u, conjugate(v))
    F = form(a, (conjugate(v), u))
    check("bilinear form, action at 1 then 0", action(action(F, 1, w), 0, z))
    check("action at 0 leaves a linear form", action(F, 0, z))
    check("sum and product of forms", action(add(multiply(num(3), F), F), 1, w))
    check(
        "product of two linear forms",
        action(
            multiply(form(multiply(v, num(2)), (v,)), form(conjugate(u), (conjugate(u),))), 1, w
        ),
    )
    check("conjugate of a form", action(conjugate(F), 0, z))
    check("pairing remains", action(form(multiply(action(l, 0, v), u), (v, u)), 0, w))

    # The family index i of g must not be captured by the ComponentTensor that binds i.
    c = tensor((num(1), num(2)))
    g = multiply(indexed(c, (i,)), w)
    G = component_tensor(
        multiply(indexed(tensor((num(5), num(7))), (i,)), form(multiply(v, num(1)), (v,))), (i,)
    )
    for k in range(2):
        check(f"capture, family index i = {k}", action(G, 0, g), {i: k})

    # The argument contains the variable of the derivative: expect h * u, not 2 h u.
    J = form(directional_derivative(multiply(u, v), u, h, u), (v,))
    out = check("action on a form with a derivative of u at u", action(J, 0, u))
    assert np.isclose(evaluate(out, env)(()), env[h] * env[u])
