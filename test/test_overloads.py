import ast
from importlib.util import resolve_name
from pathlib import Path

import numpy as np
import pytest

import uflx_spec_interpreter
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


# The method-to-factory mapping is the Python spelling of the spec's overload
# column. Arguments are part of the contract: subtraction must not swap operands.
DELEGATES = {
    "__add__": ("arithmetic.add", "self, other"),
    "__sub__": ("arithmetic.subtract", "self, other"),
    "__mul__": ("arithmetic.multiply", "self, other"),
    "__truediv__": ("arithmetic.divide", "self, other"),
    "__pow__": ("arithmetic.power", "self, other"),
    "__matmul__": ("tensors.matrix_product", "self, other"),
    "__neg__": ("arithmetic.negate", "self"),
    "__abs__": ("arithmetic.absolute", "self"),
    "__getitem__": ("tensors.component", "self, argument_members(entries)"),
}


def delegates_to(method, factory, arguments):
    """Check the non-NotImplemented return is exactly the delegated call."""
    imports = {}
    for node in ast.walk(method):
        if isinstance(node, ast.ImportFrom):
            origin = (
                resolve_name(
                    "." * node.level + (node.module or ""),
                    "uflx_spec_interpreter._src.nodes",
                )
                if node.level
                else node.module
            )
            for name in node.names:
                imports[name.asname or name.name] = f"{origin}.{name.name}"
    returns = [
        node.value
        for node in ast.walk(method)
        if isinstance(node, ast.Return)
        and not (isinstance(node.value, ast.Name) and node.value.id == "NotImplemented")
    ]
    if len(returns) != 1 or not isinstance(returns[0], ast.Call):
        return False
    call = returns[0]
    if not isinstance(call.func, ast.Name):
        return False
    if imports.get(call.func.id) != f"uflx_spec_interpreter._src.{factory}":
        return False
    expected = ast.parse(f"delegate({arguments})", mode="eval").body
    return not call.keywords and [ast.dump(a) for a in call.args] == [
        ast.dump(a) for a in expected.args
    ]


def test_operator_methods_delegate_to_the_expected_factories():
    path = Path(uflx_spec_interpreter.__file__).parent / "_src/nodes/expressions.py"
    tree = ast.parse(path.read_text())
    expression = next(
        n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "AbstractExpression"
    )
    methods = {n.name: n for n in expression.body if isinstance(n, ast.FunctionDef)}
    for name, (factory, arguments) in DELEGATES.items():
        assert name in methods, f"missing operator {name}"
        assert delegates_to(methods[name], factory, arguments), f"{name} must delegate to {factory}"


@pytest.mark.parametrize(
    ("body", "valid"),
    [
        ("from ..arithmetic import subtract; return subtract(self, other)", True),
        ("from ..arithmetic import subtract as f; return f(self, other)", True),
        ("from ..arithmetic import add; return add(self, other)", False),
        ("from ..arithmetic import subtract; return subtract(other, self)", False),
        ("from ..arithmetic import subtract; return -subtract(self, other)", False),
        ("from .arithmetic import Sum; return Sum(self, other)", False),
    ],
)
def test_delegation_check_detects_wrong_factory_and_operands(body, valid):
    method = ast.parse(f"def __sub__(self, other):\n    {body}").body[0]
    assert delegates_to(method, "arithmetic.subtract", "self, other") == valid
