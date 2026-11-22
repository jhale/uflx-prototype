from uflx._src.nodes import Complex
from uflx._src.passes.derivatives import apply_derivatives
from uflx._src.passes.gradients import apply_gradients
from uflx.arithmetic import multiply, num
from uflx.calculus import directional_derivative, gradient
from uflx.forms import function
from uflx.nodes import Integer, ScalarType
from uflx_basix.spaces import space

from ..testutil import keeps_static_type
from .test_integrals import D, K
from .test_integrals import u as on_D

V = space("V")
u, h = function(V, "u"), function(V, "h")


def test_a_zero_preserves_complex_type_but_may_lose_support():
    e = directional_derivative(multiply(Complex(Integer(0), Integer(1)), on_D), u, u, h)
    out = apply_derivatives(e)
    assert (e.scalar_type, e.domain, out.scalar_type, out.domain) == (
        ScalarType.complex,
        D,
        ScalarType.complex,
        None,
    )
    assert keeps_static_type(e, out)
    g = gradient(Complex(Integer(0), Integer(1)), K)
    assert keeps_static_type(g, apply_gradients(g))


def test_the_order_does_not_go_the_other_way():
    assert not keeps_static_type(num(1), Complex(Integer(0), Integer(1)))
    assert not keeps_static_type(num(1), on_D)
