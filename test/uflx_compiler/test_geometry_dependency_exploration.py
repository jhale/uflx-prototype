"""Geometry exposure probes against finite differences and independent field data."""

import numpy as np
import pytest

from uflx.calculus import directional_derivative
from uflx.passes import apply_derivatives

from ..uflx_basix.shape_derivatives import functionals, v0, v1, x0, x1
from .test_shape_derivatives import COORDINATES, difference, value


@pytest.mark.parametrize("name", tuple(functionals()))
def test_geometry_derivatives_agree_with_finite_differences(name):
    f = functionals()[name]
    result = apply_derivatives(directional_derivative(f, (x0, x1), (x0, x1), (v0, v1)))
    assert np.isclose(value(result, COORDINATES), difference(f), rtol=1e-6, atol=1e-8)
