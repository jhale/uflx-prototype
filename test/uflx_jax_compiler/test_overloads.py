# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import numpy as np
import pytest

from uflx.arithmetic import imaginary_part, real_part
from uflx_jax_compiler import trace

from ..uflx.test_overloads import RNG, A, B, a, array, x, y

Z = RNG.normal(size=(2, 2)) + 1j * RNG.normal(size=(2, 2))


def value(e):
    return np.asarray(trace(e)())


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
