import numpy as np
import pytest

from uflx._src.extensions.external_operators import apply_external_derivatives, external_operator
from uflx._src.nodes import AbstractFunctionSpace, Apply, Conditional
from uflx.arithmetic import conditional, less_than, num, square_root
from uflx.calculus import coordinate, directional_derivative, gradient
from uflx.forms import apply, form, function, partial
from uflx.geometry import domain_map, pull_back
from uflx.nodes import Boolean, EuclideanSpace
from uflx.passes import expand_applications, lower
from uflx.tensors import component_tensor, index, index_sum, tensor
from uflx_basix.spaces import space
from uflx_oracle import at, evaluate

from ..external_operators import Energy, EnergyTangent


def value(expression, env=None, indices=None):
    return evaluate(expression, env or {}, indices)(())


def test_boolean_values_and_lazy_selection():
    x = Boolean(True)
    assert index(("x", True)) != index(("x", 1))
    with pytest.raises(TypeError):
        bool(x)
    with pytest.raises(TypeError):
        Boolean(1)
    for make in (
        lambda: x + x,
        lambda: form(x, ()),
        lambda: tensor((x, num(1))),
        lambda: conditional(num(0), x, x),
        lambda: less_than(x, x),
    ):
        with pytest.raises(ValueError):
            make()
    i = index("i")
    p = tensor((x, Boolean(False)))[i]
    with pytest.raises(ValueError):
        index_sum(p, (i,))
    singular = num(1) / num(0)
    selected = conditional(p, num(7), singular)
    assert value(lower(selected), indices={i: 0}) == 7
    flags = component_tensor(conditional(p, p, x), (i,))
    assert np.array_equal(value(flags), [True, True])
    assert value(flags).dtype == bool
    with np.errstate(invalid="ignore"):
        real_branch = conditional(x, square_root(num(-1)), num(1j))
        assert np.isnan(value(real_branch).real)
        assert value(real_branch).imag == 0


def test_conditional_derivatives_and_pullback():
    P = EuclideanSpace(1)
    x = coordinate(P)[0]
    f = conditional(less_than(x, num(0)), x * x, x)
    g = lower(gradient(f, P))
    for point, expected in [(-2.0, -4.0), (2.0, 1.0)]:
        assert np.allclose(at(P, np.array([point]), lambda: value(g)), [expected])
    phi = domain_map(P, tensor((num(-2),)), P)
    assert value(lower(pull_back(f, phi))) == 4
    V = space("R", scalar_type="real")
    v = function(V, "v")
    f = conditional(less_than(v, num(0)), v * v, v)
    assert value(lower(directional_derivative(f, v, num(-2), num(3)))) == -12


def test_full_application_and_family_completion():
    V = space("R", scalar_type="real")
    x, y = (function(V, n) for n in ("x", "y"))
    f = form(x * y, (x, y))
    call = apply(f, (y, x))
    assert isinstance(call, Apply) and call.arguments == (y, x)
    assert expand_applications(call) is y * x
    flag = Boolean(False)
    result = lower(apply(conditional(flag, f, f + f), (x, y)))
    assert isinstance(result, Conditional)
    assert value(result, {x: 3, y: 5}) == value(apply(f + f, (y, x)), {x: 3, y: 5}) == 30
    i = index("unused")
    data = tensor((num(2), num(3)))[i]
    for result in (
        partial(form(x, (x, y)), {y: data}),
        expand_applications(apply(form(x, (y,)), (data,))),
    ):
        assert result.free_indices == ((i, 2),)
    with pytest.raises(ValueError):
        Apply(x, ())


def test_derivative_of_full_terminal_application():
    V = space("R", scalar_type="real")
    x = function(V, "x")
    # Distinct input names declare an ordered bilinear map.
    y = function(V, "y")
    F = form(x * y, (x, y))

    class BilinearSpace(AbstractFunctionSpace):
        identifier = "test.BilinearSpace"
        value_shape = ()
        domain = None
        scalar_type = "real"
        argument_spaces = (V, V)

    A = function(BilinearSpace(), "A")
    expression = apply(A, (x, x))
    env = {A: lambda a, b: 2 * a * b}
    assert value(lower(directional_derivative(expression, x, num(3), num(1))), env) == 12
    assert value(lower(directional_derivative(expression, A, F, F)), {x: 3}) == 9


def test_external_operator_derivative_action_and_missing_rule():
    x = function(space("R", scalar_type="real"), "x")
    f = external_operator(Energy(), (x, x))
    d = apply_external_derivatives(directional_derivative(f, x, x, num(3)))
    assert d.operation is EnergyTangent() and d.operands == (x, x, num(3), num(3))
    assert value(d, {x: 2, EnergyTangent(): lambda a, b, h, k: h * b + a * k}) == 12
    with pytest.raises(TypeError, match="no derivative action"):
        apply_external_derivatives(directional_derivative(d, x, x, num(1)))
