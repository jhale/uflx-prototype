# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import runpy
from pathlib import Path

import basix
import jax
import numpy as np
import pytest

import uflx_jax_compiler
from uflx.arithmetic import num
from uflx.calculus import directional_derivative, gradient, integral
from uflx.forms import function
from uflx.geometry import create_mesh, euclidean_space
from uflx.passes import lower
from uflx.tensors import index
from uflx_basix.cells import cell, topology
from uflx_basix.evaluation import evaluate_at_points
from uflx_basix.spaces import local_space
from uflx_extensions.runtime_arrays import runtime_array, runtime_extent
from uflx_jax_compiler import _finite_elements, reference_trace


def test_fused_geometry_and_assembly_use_fixed_basix_tables(monkeypatch):
    def unexpected(*args, **kwargs):
        pytest.fail("fixed-point stiffness should use Basix derivative rows")

    with monkeypatch.context() as patch:
        patch.setattr(jax, "jacfwd", unexpected)
        patch.setattr(jax, "jvp", unexpected)
        patch.setattr(_finite_elements, "tabulate", unexpected)
        example = Path(__file__).resolve().parents[2] / "examples" / "precomputed_geometry.py"
        demo = runpy.run_path(str(example))
        kernel = demo["cell_stiffness"]
        coordinates = demo["coordinates"]
        compiled = kernel.lower(coordinates).compile()
        reflected = compiled(coordinates[:, ::-1])
    points, weights = demo["points"], demo["weights"]
    basis = demo["p2"].tabulate(1, points)[1:, :, :, 0]
    jacobians = np.einsum("rqi,ia->qar", basis, coordinates)
    gradients = np.einsum("rqi,qra->qia", basis, np.linalg.inv(jacobians))
    expected = np.einsum(
        "q,qia,qja->ij", weights * abs(np.linalg.det(jacobians)), gradients, gradients
    )
    np.testing.assert_allclose(demo["matrix"], expected, atol=1e-13)
    np.testing.assert_allclose(reflected, expected, atol=1e-13)

    direction = np.sin(coordinates)
    tangent = jax.jvp(kernel, (coordinates,), (direction,))[1]
    h = 1e-5
    difference = (compiled(coordinates + h * direction) - compiled(coordinates - h * direction)) / (
        2 * h
    )
    np.testing.assert_allclose(tangent, difference, rtol=1e-7, atol=1e-9)


@pytest.mark.parametrize("build", [uflx_jax_compiler.trace, reference_trace])
def test_standalone_samples_read_supplied_basix_partials(build):
    K = cell(basix.CellType.triangle)
    p2 = basix.create_element(basix.ElementFamily.P, K.cell_type, 2)
    u = function(local_space(K, p2, shape=(2,)), "u")
    points = runtime_array("points", (runtime_extent("n"), 2))
    samples = evaluate_at_points(gradient(gradient(u, K), K), points, index("q"))
    (table,) = uflx_jax_compiler.basis_tables(samples)
    assert table.order == 2 and table.element == p2 and table.point_source is points
    kernel = jax.jit(build(samples, points, table, u))
    rng = np.random.default_rng(42)
    coefficients = rng.normal(size=(p2.dim, 2))
    for n in (0, 3, 6):
        pts = rng.uniform(0, 0.5, size=(n, 2))
        rows = rng.normal(size=(6, n, p2.dim, 1))
        hessian = rows[
            [[basix.index(2, 0), basix.index(1, 1)], [basix.index(1, 1), basix.index(0, 2)]]
        ]
        expected = np.einsum("abqi,ic->qcab", hessian[..., 0], coefficients)
        np.testing.assert_allclose(kernel(pts, rows, coefficients), expected, atol=1e-13)
    if build is uflx_jax_compiler.trace:
        first = evaluate_at_points(gradient(u, K), points, index("q"))
        with pytest.raises(ValueError, match="Basix derivative table"):
            uflx_jax_compiler.trace(first, points, u)(pts, coefficients)
        with pytest.raises(ValueError, match="supplied field"):
            uflx_jax_compiler.trace(samples, points, table, u)(pts, rows, lambda point: point**2)


def test_trace_requires_lowering_and_reference_trace_accepts_unlowered_forms():
    T = topology(basix.CellType.interval)
    p1 = basix.create_element(basix.ElementFamily.P, basix.CellType.interval, 1)
    x = function(local_space(T, p1, shape=(1,)), "x")
    mesh = create_mesh(x, euclidean_space(1))
    volume = integral(num(1), mesh)
    rule = basix.make_quadrature(basix.CellType.interval, 2)
    with pytest.raises(ValueError, match=r"Physical mesh integration.*requires lowering"):
        uflx_jax_compiler.trace(volume, x, volume)(p1.points, rule)
    volume_reference = reference_trace(volume, x, volume)
    np.testing.assert_allclose(volume_reference(p1.points, rule), 1.0)
    derivative = directional_derivative(x, x, x, x)
    with pytest.raises(TypeError, match="allow_ad"):
        uflx_jax_compiler.trace(derivative, x, T, allow_ad=True)
    with pytest.raises(ValueError, match=r"DirectionalDerivative.*requires lowering"):
        uflx_jax_compiler.trace(derivative, x, T)(p1.points, [0.25])
    evaluate = uflx_jax_compiler.trace(lower(derivative), x, T)
    reference = reference_trace(derivative, x, T)
    np.testing.assert_allclose(evaluate(p1.points, [0.25]), reference(p1.points, [0.25]))
