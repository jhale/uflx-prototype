import numpy as np

from uflx_spec_interpreter.arithmetic import conjugate, multiply
from uflx_spec_interpreter.calculus import directional_derivative
from uflx_spec_interpreter.forms import function, space
from uflx_spec_interpreter.indices import ellipsis_index, index
from uflx_spec_interpreter.literals import num
from uflx_spec_interpreter.oracle import evaluate
from uflx_spec_interpreter.passes import derivatives
from uflx_spec_interpreter.substitution import substitute
from uflx_spec_interpreter.tensors import component_tensor, index_sum, indexed, tensor

r, s = ellipsis_index("r"), ellipsis_index("s")
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


def test_an_ellipsis_covers_the_axes_that_the_other_entries_leave():
    t = array(T)
    assert indexed(t, (r,)).free_indices == ((r, (2, 3, 2)),)
    assert indexed(t, (0, r)).free_indices == ((r, (3, 2)),)
    assert indexed(t, (r, 1)).free_indices == ((r, (2, 3)),)
    assert indexed(t, (i, r, k)).free_indices == ((i, 2), (r, (3,)), (k, 2))
    assert indexed(t, (i, 0, k, r)).free_indices == ((i, 2), (k, 2), (r, ()))
    scaled = component_tensor(multiply(num(2.0), indexed(t, (r,))), (r,))
    assert scaled.value_shape == (2, 3, 2) and scaled.free_indices == ()
    assert index_sum(indexed(t, (r,)), (r,)).value_shape == ()


def test_the_compositions_of_the_examples_have_the_values_of_numpy():
    u, v, a, b, t = array(U), array(V), array(A), array(B), array(T)
    outer = component_tensor(multiply(indexed(u, (r,)), indexed(v, (s,))), (r, s))
    assert np.allclose(value(outer), np.outer(U, V))
    dot = component_tensor(
        index_sum(multiply(indexed(a, (r, k)), indexed(b, (k, s))), (k,)), (r, s)
    )
    assert np.allclose(value(dot), A @ B)
    vector_dot = component_tensor(
        index_sum(multiply(indexed(u, (r, k)), indexed(a, (k, s))), (k,)), (r, s)
    )
    assert vector_dot.value_shape == (3,) and np.allclose(value(vector_dot), U @ A)
    inner = index_sum(multiply(indexed(t, (r,)), conjugate(indexed(t, (r,)))), (r,))
    assert np.allclose(value(inner), np.sum(T * T))
    same = component_tensor(indexed(u, (i, r)), (i, r))
    assert same.value_shape == (2,) and np.allclose(value(same), U)


def test_a_directional_derivative_through_an_ellipsis():
    x, h = function(space("V"), "x"), function(space("V"), "h")
    F = component_tensor(multiply(x, multiply(x, indexed(array(T), (r,)))), (r,))
    f = index_sum(multiply(indexed(F, (r,)), indexed(F, (r,))), (r,))
    e = directional_derivative(f, x, h)
    out = derivatives.apply_derivatives(e)
    env = {x: 0.7, h: 0.3}
    assert np.isclose(evaluate(e, env)(()), evaluate(out, env)(()), rtol=1e-6)
    assert np.isclose(evaluate(out, env)(()), 4 * 0.7**3 * 0.3 * np.sum(T * T))


def test_substitution_renames_a_bound_ellipsis():
    # The replacement brings a free r, which the binder of r must not capture.
    y = function(space("V"), "y")
    e = component_tensor(multiply(y, indexed(array(T), (r,))), (r,))
    out = substitute(e, {y: indexed(array(U), (r,))})
    assert out.value_shape == (2, 3, 2) and out.free_indices == ((r, (2,)),)
    for m in range(2):
        assert np.allclose(value(out, {r: (m,)}), U[m] * T)


def test_a_gradient_of_a_tensor_built_with_an_ellipsis():
    from test_integrals import K

    from uflx_spec_interpreter.calculus import coordinate, gradient, integral
    from uflx_spec_interpreter.passes import gradients

    X = coordinate(K)
    xy = multiply(indexed(X, (0,)), indexed(X, (1,)))
    f = component_tensor(multiply(xy, indexed(array(T), (r,))), (r,))
    e = integral(indexed(gradient(f, K), (1, 2, 0, 1)), K)
    out = gradients.apply_gradients(e)
    assert gradients.in_n6(out) and not gradients.in_n6(e)
    assert np.isclose(evaluate(e, {})(()), evaluate(out, {})(()), rtol=1e-6)
