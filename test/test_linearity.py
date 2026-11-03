"""Shared certificates and their use by derivative lowering."""

import pytest

from uflx_spec_interpreter import (
    Integer,
    action,
    conjugate,
    dual_space,
    form,
    function,
    gateaux_derivative,
    num,
    tensor,
)
from uflx_spec_interpreter._src.linearity import _argument_dependence, _dependence
from uflx_spec_interpreter._src.nodes import Complex, DirectionalDerivative, RealPart
from uflx_spec_interpreter._src.passes import derivatives
from uflx_toy.spaces import space


@pytest.mark.parametrize(
    ("build", "expected"),
    [
        (lambda u, v: num(0), "zero"),
        (lambda u, v: u, "complex-linear"),
        (lambda u, v: num(3) * u, "complex-linear"),
        (lambda u, v: conjugate(u), "complex-antilinear"),
        (lambda u, v: u + conjugate(u), "real-linear"),
        (lambda u, v: RealPart(u), "real-linear"),
        (lambda u, v: u + v, "affine"),
        (lambda u, v: u * u, "unknown"),
        (lambda u, v: u * conjugate(u), "unknown"),
        (lambda u, v: u / v, "complex-linear"),
        (lambda u, v: v / u, "unknown"),
        (lambda u, v: tensor((u, conjugate(u))), "real-linear"),
    ],
)
def test_dependence(build, expected):
    V = space("linearity")
    u, v = function(V, "u"), function(V, "v")
    assert _dependence(build(u, v), u) == expected


def test_joint_dependence_and_argument_conventions():
    V = space("joint")
    u, v = function(V, "u"), function(V, "v")
    assert _dependence(u + v, (u, v)) == "complex-linear"
    assert _dependence(u * v, u) == "complex-linear"
    assert _dependence(u * v, (u, v)) == "unknown"
    F = form(u * conjugate(v), (u, conjugate(v)))
    assert _argument_dependence(F, 0) == "complex-linear"
    assert _argument_dependence(F, 1) == "complex-antilinear"
    assert _dependence(F, u) == "independent"
    assert _dependence(form(u * v, (v,)), u) == "complex-linear"


def test_opaque_linear_action_and_unrecognised_operation():
    from test_derivative_algebra import LibraryValue

    V = space("opaque-linearity")
    u = function(V, "u")
    L = function(dual_space(V), "L")
    assert _dependence(action(L, 0, u), u) == "complex-linear"
    assert _dependence(LibraryValue(u), u) == "unknown"


@pytest.mark.parametrize("antilinear", [False, True])
def test_rotated_direction_uses_the_general_differential(monkeypatch, antilinear):
    V = space("reuse")
    u, h = function(V, "u"), function(V, "h")
    body = conjugate(u * u) if antilinear else u * u
    calls = []
    original = derivatives._deriv

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(derivatives, "_deriv", record)
    out = derivatives.apply_derivatives(gateaux_derivative(body, u, u, h))
    assert len(calls) == 2
    from uflx_spec_interpreter._src.passes.algebra import _normalizer

    assert _argument_dependence(_normalizer()(out), 0) in ("zero", "complex-linear")


def test_direction_occurring_at_point_does_not_use_shortcut(monkeypatch):
    V = space("nonfresh-direction")
    u, h = function(V, "u"), function(V, "h")
    calls = []
    original = derivatives._deriv

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(derivatives, "_deriv", record)
    derivatives.apply_derivatives(
        DirectionalDerivative(u * u, u, h, Complex(Integer(0), Integer(1)) * h)
    )
    assert len(calls) == 1
    assert calls[0][3] is Complex(Integer(0), Integer(1)) * h


def test_real_variables_are_reported_as_real_linear():
    from uflx_spec_interpreter import ScalarType

    u = function(space("real-linearity", scalar_type=ScalarType.real), "u")
    assert _dependence(u, u) == "real-linear"
    assert _dependence(conjugate(u), u) == "real-linear"
    with pytest.raises(TypeError):
        _dependence(u, ())


def test_mixed_differential_retains_general_projection(monkeypatch):
    V = space("mixed-fallback")
    u, h = function(V, "u"), function(V, "h")
    calls = []
    original = derivatives._deriv

    def record(*args, **kwargs):
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(derivatives, "_deriv", record)
    derivatives.apply_derivatives(gateaux_derivative(u * conjugate(u), u, u, h))
    assert len(calls) == 2


def test_certificates_obey_complex_scaling_and_addition():
    import numpy as np

    from uflx_oracle import evaluate

    V = space("certificate-values")
    u, a = function(V, "u"), function(V, "a")
    expressions = (
        a * u,
        a * conjugate(u),
        a * u + conjugate(u),
        RealPart(a * u),
        tensor((a * u, u)),
    )
    z, w, c = 0.3 + 0.7j, -0.2 + 0.1j, 1.2 - 0.8j
    for expression in expressions:
        kind = _dependence(expression, u)

        def value(z):
            return np.asarray(evaluate(expression, {u: z, a: 0.6 - 0.2j})(()))

        assert np.allclose(value(z + w), value(z) + value(w))
        assert np.allclose(value(2 * z), 2 * value(z))
        if kind == "complex-linear":
            assert np.allclose(value(c * z), c * value(z))
        elif kind == "complex-antilinear":
            assert np.allclose(value(c * z), c.conjugate() * value(z))
        else:
            assert kind == "real-linear"


@pytest.mark.parametrize("position", [-1, 1, True, 0.0])
def test_argument_query_rejects_invalid_positions(position):
    u = function(space("query-position"), "u")
    error = TypeError if type(position) is not int else ValueError
    with pytest.raises(error):
        _argument_dependence(form(u, (u,)), position)


def test_query_input_validation_and_inherited_argument_positions():
    u, v = (function(space("query-contract"), label) for label in ("u", "v"))
    with pytest.raises(ValueError, match="distinct"):
        _dependence(u, (u, u))
    assert _dependence(u, (u,)) == "complex-linear"
    nested = form(form(u * conjugate(v), (u,)), (conjugate(v),))
    assert _argument_dependence(nested, 0) == "complex-antilinear"
    with pytest.raises(ValueError):
        _argument_dependence(nested, 1)
