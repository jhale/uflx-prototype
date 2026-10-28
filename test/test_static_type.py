from test_integrals import D, K
from test_integrals import u as on_D
from testutil import keeps_static_type

from uflx_spec_interpreter import directional_derivative, function, gradient, multiply, num
from uflx_spec_interpreter._src.nodes import ImaginaryUnit
from uflx_spec_interpreter._src.passes.derivatives import apply_derivatives
from uflx_spec_interpreter._src.passes.gradients import apply_gradients
from uflx_toy.spaces import space

V = space("V")
u, h = function(V, "u"), function(V, "h")


def test_a_zero_may_make_a_complex_field_real_and_constant():
    e = directional_derivative(multiply(ImaginaryUnit(), on_D), u, u, h)
    out = apply_derivatives(e)
    assert (e.scalar_type, e.support, out.scalar_type, out.support) == ("complex", D, "real", None)
    assert keeps_static_type(e, out)
    g = gradient(ImaginaryUnit(), K)
    assert keeps_static_type(g, apply_gradients(g))


def test_the_order_does_not_go_the_other_way():
    assert not keeps_static_type(num(1), ImaginaryUnit())
    assert not keeps_static_type(num(1), on_D)
