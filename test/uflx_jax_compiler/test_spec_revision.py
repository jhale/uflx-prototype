"""Regression examples in the order of the revised language chapters."""

from __future__ import annotations

import numpy as np
import pytest

from uflx._src.nodes import Determinant
from uflx._src.nodes.structure import expression_operands
from uflx.arithmetic import conjugate
from uflx.calculus import coordinate, directional_derivative, gradient
from uflx.forms import apply, form, function
from uflx.nodes import Complex, EuclideanSpace, Float, Integer, ScalarType, Zero
from uflx.passes import lower
from uflx.tensors import determinant
from uflx_jax_compiler import trace

from ..uflx.library import runtime_array, runtime_extent
from ..uflx.spaces import space


def test_complex_is_a_terminal_and_stays_complex():
    z = Complex(Integer(2), Float(-3.0))
    assert expression_operands(z) == []
    assert z.scalar_type is ScalarType.complex
    assert trace(z)() == 2 - 3j
    assert lower(Complex(Integer(1), Integer(0))).scalar_type is ScalarType.complex


def test_determinants_require_fixed_dimensions_and_accept_empty_shapes():
    n = runtime_extent("n")
    with pytest.raises(ValueError, match="fixed integers"):
        determinant(runtime_array("A", (n, n)))
    assert isinstance(determinant(runtime_array("A", (2, 2))), Determinant)
    d = determinant(Zero((0, 0), (), (), ScalarType.real))
    assert lower(d) is d
    assert trace(d)() == 1
    with pytest.raises(ValueError):
        determinant(runtime_array("rectangular", (2, 3)))


def test_action_conjugation_and_real_to_complex_embedding():
    V = space("V")
    u = function(V, "u")
    e = apply(conjugate(form(u, (u,))), (Complex(Integer(2), Integer(3)),))
    assert trace(lower(e))() == 2 + 3j
    f = apply(form(u ** Float(0.5), (u,)), (Integer(-1),))
    assert lower(f).scalar_type is ScalarType.complex
    assert np.isclose(trace(lower(f))(), 1j)


def test_derivative_substitutes_the_point_but_leaves_the_direction_alone():
    V = space("R", scalar_type=ScalarType.real)
    u = function(V, "u")
    e = directional_derivative(u * u, u, Integer(3), u)
    assert np.isclose(trace(lower(e), u)(2), 12)


def test_zero_dimensional_coordinate_gradient_has_empty_shape():
    e = gradient(coordinate(EuclideanSpace(0)), EuclideanSpace(0))
    out = lower(e)
    assert out.value_shape == (0, 0)
    assert trace(out)().shape == (0, 0)
