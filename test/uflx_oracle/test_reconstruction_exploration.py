"""Physical Hermite degrees of freedom require explicit geometry reconstruction.

This is one affine interval, not a general Hermite element implementation.
The four independent data are endpoint values and physical endpoint slopes.
"""

import numpy as np

from uflx.algorithms import replace
from uflx.arithmetic import num
from uflx.calculus import directional_derivative
from uflx.forms import function
from uflx.passes import apply_derivatives
from uflx_basix.spaces import space
from uflx_oracle.interpreter import Interpreter

V = space("Hermite-independent-data", scalar_type="real")
length, s, q0, m0, q1, m1 = (function(V, name) for name in ("L", "s", "q0", "m0", "q1", "m1"))
# x = L*s. Reference slopes must be L times the supplied physical slopes.
p = (
    (num(2) * s ** num(3) - num(3) * s ** num(2) + num(1)) * q0
    + length * (s ** num(3) - num(2) * s ** num(2) + s) * m0
    + (num(-2) * s ** num(3) + num(3) * s ** num(2)) * q1
    + length * (s ** num(3) - s ** num(2)) * m1
)
DATA = {length: 1.7, s: 0.37, q0: 0.4, m0: -0.2, q1: 1.3, m1: 0.8}


def scalar(e, data):
    return Interpreter(data).evaluate(e)


def test_reconstruction_preserves_physical_endpoint_dofs():
    physical_slope = apply_derivatives(directional_derivative(p, s, s, num(1))) / length
    for endpoint, value, slope in ((0, q0, m0), (1, q1, m1)):
        data = DATA | {s: endpoint}
        assert np.isclose(scalar(p, data), DATA[value])
        assert np.isclose(scalar(physical_slope, data), DATA[slope])


def test_geometry_and_mixed_derivatives_include_reconstruction():
    f = p * p
    dL = apply_derivatives(directional_derivative(f, length, length, num(1)))
    mixed = apply_derivatives(directional_derivative(dL, m0, m0, num(1)))
    reverse = apply_derivatives(
        directional_derivative(
            apply_derivatives(directional_derivative(f, m0, m0, num(1))), length, length, num(1)
        )
    )
    assert np.isclose(scalar(mixed, DATA), scalar(reverse, DATA))
    step = 1e-5
    difference = (
        scalar(f, DATA | {length: DATA[length] + step})
        - scalar(f, DATA | {length: DATA[length] - step})
    ) / (2 * step)
    assert abs(difference) > 1e-3
    assert np.isclose(scalar(dL, DATA), difference, rtol=1e-8)
    # Freezing the reconstruction matrix loses a real geometry derivative.
    frozen = replace(f, {length: num(DATA[length])})
    assert (
        scalar(apply_derivatives(directional_derivative(frozen, length, length, num(1))), DATA) == 0
    )
