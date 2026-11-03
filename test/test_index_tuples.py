import numpy as np

from uflx_oracle import evaluate
from uflx_spec_interpreter import (
    component_tensor,
    directional_derivative,
    function,
    index,
    index_sum,
    indexed,
    multiply,
    num,
    tensor,
)
from uflx_spec_interpreter._src.contracts import _in_core_normal_form
from uflx_spec_interpreter._src.passes import derivatives
from uflx_spec_interpreter._src.substitution import _substitute
from uflx_toy.spaces import space

r = tuple(index(f"r{n}") for n in range(3))
s = tuple(index(f"s{n}") for n in range(3))
i, k = index("i"), index("k")


def array(values):
    # A tensor of literals with the values of a NumPy array.
    values = np.asarray(values)
    if values.ndim == 0:
        return num(float(values))
    return tensor(tuple(array(v) for v in values))


RNG = np.random.default_rng(0)
U, V = RNG.normal(size=2), RNG.normal(size=3)
A, B, T = RNG.normal(size=(2, 3)), RNG.normal(size=(3, 2)), RNG.normal(size=(2, 3, 2))


def value(e, ienv=None):
    return np.real(np.asarray(evaluate(e, {}, ienv)(())))


def test_one_index_per_axis_retains_extents_in_order():
    t = array(T)
    assert indexed(t, r).free_indices == tuple(zip(r, T.shape, strict=True))
    scaled = component_tensor(num(2) * indexed(t, r), r)
    assert scaled.value_shape == T.shape and scaled.free_indices == ()
    assert np.allclose(value(scaled), 2 * T)
    assert np.isclose(value(index_sum(indexed(t, r), r)), T.sum())


def test_scalar_empty_index_identities():
    scalar = num(2)
    assert indexed(scalar, ()) is scalar
    assert component_tensor(scalar, ()) is scalar
    assert index_sum(scalar, ()) is scalar


def test_a_directional_derivative_through_index_tuples():
    x, h = function(space("V"), "x"), function(space("V"), "h")
    F = component_tensor(multiply(x, multiply(x, indexed(array(T), r))), r)
    f = index_sum(multiply(indexed(F, r), indexed(F, r)), r)
    e = directional_derivative(f, x, x, h)
    out = derivatives.apply_derivatives(e)
    env = {x: 0.7, h: 0.3}
    assert np.isclose(evaluate(e, env)(()), evaluate(out, env)(()), rtol=1e-6)
    assert np.isclose(evaluate(out, env)(()), 4 * 0.7**3 * 0.3 * np.sum(T * T))


def test_substitution_renames_a_bound_index_tuple():
    # The replacement brings a free r, which the binder of r must not capture.
    y = function(space("V"), "y")
    e = component_tensor(multiply(y, indexed(array(T), r)), r)
    out = _substitute(e, {y: indexed(array(U), (r[0],))})
    assert out.value_shape == (2, 3, 2) and out.free_indices == ((r[0], 2),)
    for m in range(2):
        assert np.allclose(value(out, {r[0]: m}), U[m] * T)


def test_a_gradient_of_a_tensor_built_with_index_tuples():
    from test_integrals import K

    from uflx_spec_interpreter import coordinate, gradient, integral
    from uflx_spec_interpreter._src.passes import gradients

    X = coordinate(K)
    xy = multiply(indexed(X, (0,)), indexed(X, (1,)))
    f = component_tensor(multiply(xy, indexed(array(T), r)), r)
    e = integral(indexed(gradient(f, K), (1, 2, 0, 1)), K)
    out = gradients.apply_gradients(e)
    assert _in_core_normal_form(out) and not _in_core_normal_form(e)
    assert np.isclose(evaluate(e, {})(()), evaluate(out, {})(()), rtol=1e-6)
