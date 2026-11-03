import numpy as np
import pytest

from uflx_oracle import evaluate
from uflx_spec_interpreter import imaginary_part, index, num, real_part, tensor
from uflx_spec_interpreter._src.nodes import ComponentTensor, Indexed, Integer, Power, Product, Sum

RNG = np.random.default_rng(1)
A, B = RNG.normal(size=(2, 3)), RNG.normal(size=(3, 3))
x, y = RNG.normal(size=3), RNG.normal(size=3)
Z = RNG.normal(size=(2, 2)) + 1j * RNG.normal(size=(2, 2))


def array(values):
    values = np.asarray(values)
    if values.ndim == 0:
        return num(values.item())
    return tensor(tuple(array(v) for v in values))


def value(e):
    return np.asarray(evaluate(e, {}, None)(()))


a = num(1.5)


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda: array(A)[1], A[1]),
        (lambda: array(A)[1, 2], A[1, 2]),
        (lambda: array(A) + array(A), A + A),
        (lambda: array(A) - array(A) * a, A - A * 1.5),
        (lambda: a * array(A), 1.5 * A),
        (lambda: array(A) / a, A / 1.5),
        (lambda: -array(x), -x),
        (lambda: array(x) @ array(y), x @ y),
        (lambda: array(A) @ array(x), A @ x),
        (lambda: array(x) @ array(B), x @ B),
        (lambda: array(A) @ array(B), A @ B),
        (lambda: real_part(array(Z)), Z.real),
        (lambda: imaginary_part(array(Z)), Z.imag),
    ],
)
def test_overloads_agree_with_numpy(build, expected):
    e = build()
    assert e.value_shape == np.shape(expected)
    assert np.allclose(value(e), expected)


def test_an_overload_is_one_free_function_and_never_regroups():
    b, c = num(2), num(3)
    assert a + b + c == Sum(Sum(a, b), c)
    assert a + (b + c) != a + b + c
    assert a**b**c == Power(a, Power(b, c))
    assert -(a**b) == Product(Integer(-1), Power(a, b))


def test_partial_indexing_is_a_component_tensor():
    M = array(A)
    assert isinstance(M[1], ComponentTensor) and M[1].value_shape == (3,)
    assert M[1] == M[1]
    assert isinstance(M[1, 0], Indexed)


def test_python_numbers_are_rejected():
    X = array(x)
    for build in (lambda: 2 * X[0], lambda: X[0] ** 2, lambda: 1 - X[0], lambda: X[0] + 0.5):
        with pytest.raises(TypeError):
            build()


def test_expressions_are_not_iterable():
    with pytest.raises(TypeError):
        list(array(x))


def test_free_indices_of_the_operands_stay_free():
    i = index("i")
    e = array(A)[i] + array(A)[i]
    assert e.value_shape == (3,)
    assert dict(e.free_indices) == {i: 2}
