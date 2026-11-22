"""General reference-cell interpolation and its construction constraints."""

import basix
import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.forms import function
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace
from uflx.tensors import component_tensor, index, tensor
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_basix.interpolation import interpolate
from uflx_basix.spaces import space
from uflx_compiler import compile_form, evaluate

from ..uflx_basix.test_interpolation import interpolation_case, interpolation_cases


def sample(f, point, cell):
    return integral(
        pull_back(
            f, domain_map(EuclideanSpace(0), tensor(tuple(num(float(x)) for x in point)), cell)
        ),
        EuclideanSpace(0),
    )


@interpolation_cases
def test_interpolation_matches_basix_sampling(cell_type, family, degree, shape):
    K, V, f, result = interpolation_case(cell_type, family, degree, shape)
    E = V.element(0)
    point = np.full(K.dimension, 0.17)
    values = np.array([evaluate(compile_form(sample(f, p, K), 4), {}) for p in E.points])
    if shape:
        dofs = E.interpolation_matrix @ values.reshape(len(E.points), -1)
        expected = (E.tabulate(0, point[None])[0, 0, :, 0] @ dofs).reshape(shape)
    else:
        dofs = E.interpolation_matrix @ values.reshape(len(E.points), -1).T.ravel()
        expected = (dofs @ E.tabulate(0, point[None])[0, 0]).reshape(V.value_shape)
    actual = evaluate(compile_form(sample(result, point, K), 4), {})
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-12)


def test_complex_reference_cell_function_can_be_interpolated():
    K = basix_cell(basix.CellType.triangle)
    U = basix_space("source", K, basix.ElementFamily.P, 1, scalar_type="complex")
    V = basix_space("target", K, basix.ElementFamily.P, 2, scalar_type="complex")
    u = function(U, "u")
    result = interpolate(u, V)
    values = np.array([1 + 2j, 3 - 1j, -2 + 0.5j])
    point = np.array([0.2, 0.3])
    expected = U.element(0).tabulate(0, point[None])[0, 0, :, 0] @ values
    actual = evaluate(compile_form(sample(result, point, K), 4), {"coefficients:u": values})
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    assert result.scalar_type == "complex"


def test_constant_interpolation_retains_target_domain():
    K = basix_cell(basix.CellType.triangle)
    V = basix_space("constant", K, basix.ElementFamily.P, 0, discontinuous=True)
    result = interpolate(num(3), V)
    assert result.domain == K
    assert evaluate(compile_form(sample(result, (0.2, 0.3), K), 0), {}) == pytest.approx(3)


def test_interpolation_preserves_input_free_indices():
    K = basix_cell(basix.CellType.triangle)
    V = basix_space("target", K, basix.ElementFamily.P, 1)
    coefficients = function(space("coefficients", (2,), scalar_type="real"), "c")
    a = index("a")
    f = coefficients[a] * coordinate(K)[0]
    result = interpolate(f, V)
    assert result.free_indices == f.free_indices
    values = component_tensor(result, (a,))
    actual = evaluate(
        compile_form(sample(values, (0.2, 0.3), K), 2), {"coefficients:c": np.array([2.0, 3.0])}
    )
    np.testing.assert_allclose(actual, [0.4, 0.6], atol=1e-12)
