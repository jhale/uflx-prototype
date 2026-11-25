"""Evaluate a traced form directly on packed UFCx cell data."""

import basix
import jax
import numpy as np
import pytest

import uflx_jax_compiler
from uflx.calculus import integral
from uflx.forms import function
from uflx.geometry import create_mesh, euclidean_space, push_forward
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_extensions.runtime_arrays import runtime_array


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
def test_ufcx_cell_coefficients_and_constants(dtype):
    cell = basix.CellType.triangle
    T = topology(cell)
    p1 = basix.create_element(basix.ElementFamily.P, cell, 1)
    V = local_space(T, p1)
    vector_space = local_space(T, p1, shape=(2,))
    coordinates = function(vector_space, "x")
    u = function(V, "u")
    v = function(vector_space, "v")
    scale = runtime_array("scale", ())
    shift = runtime_array("shift", (2,))
    mesh = create_mesh(coordinates, euclidean_space(2))
    integrand = (
        push_forward(u, mesh, V.law) * scale + push_forward(v, mesh, vector_space.law) @ shift
    )
    expression = apply_quadrature(lower(integral(integrand, mesh)), *basix.make_quadrature(cell, 1))
    xyz = np.array([[0, 0, 12], [2, 0, 24], [0, 1, 36]], dtype=dtype)
    dofs = np.array([1, 2, 6, 1, 3, 2, 4, 3, 5], dtype=dtype)
    constants = np.array([2, 5, -1], dtype=dtype)
    evaluate = uflx_jax_compiler.trace(expression, coordinates, u, scale, v, shift)
    weighted_load = jax.jit(uflx_jax_compiler.with_ufcx_cell_inputs(evaluate, dtype=dtype))
    inputs = (xyz.ravel(), dofs, constants)
    value = weighted_load(*inputs)
    assert value.dtype == dtype
    # Unit area; means of the linear fields are 3 and (2, 4).
    np.testing.assert_allclose(value, 3 * 2 + 2 * 5 - 4)

    with pytest.raises(ValueError, match="coordinate_dofs shape"):
        weighted_load.lower(jax.ShapeDtypeStruct((3, 3), dtype), dofs, constants)
    with pytest.raises(ValueError, match="w shape"):
        weighted_load.lower(inputs[0], jax.ShapeDtypeStruct((10,), dtype), constants)
    with pytest.raises(ValueError, match="c shape"):
        weighted_load.lower(inputs[0], dofs, jax.ShapeDtypeStruct((2,), dtype))
    with pytest.raises(TypeError, match="w dtype"):
        weighted_load.lower(inputs[0], jax.ShapeDtypeStruct(dofs.shape, np.int32), constants)
