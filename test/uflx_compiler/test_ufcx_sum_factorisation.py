"""Tensor contractions, UFCx ordering and fallback for staged scalar actions."""

import ctypes
import re
import shutil
import subprocess

import basix
import numpy as np
import pytest

from uflx.arithmetic import num
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import mesh, push_forward
from uflx.nodes import EuclideanSpace
from uflx.tensors import broadcast_to, index, index_sum, tensor
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_basix.runtime import runtime_array
from uflx_compiler import compile_ufcx
from uflx_compiler.quadrature import constant
from uflx_compiler.sum_factorisation import action_kernel

from ..testutil import grouped_topology
from . import test_ufcx
from .test_ufcx import build, model

ufcx_include = test_ufcx.ufcx_include


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("shape", [(3, 4), (2, 3, 4)])
def test_directional_contractions(tmp_path, shape):
    rng = np.random.default_rng(983)
    quadrature = tuple(n + 1 for n in shape)
    q = tuple(index(f"q{k}") for k in range(len(shape)))
    i = tuple(index(f"i{k}") for k in range(len(shape)))
    a = tuple(index(f"a{k}") for k in range(len(shape)))
    basis = [rng.normal(size=(m, n)) for m, n in zip(quadrature, shape, strict=True)]
    coefficients = rng.normal(size=shape)
    weights = rng.normal(size=quadrature)
    coefficient = runtime_array("coefficients", shape)[i]
    test = num(1)
    for k, values in enumerate(basis):
        coefficient = coefficient * constant(values)[q[k], i[k]]
        test = test * constant(values)[q[k], a[k]]
    expression = index_sum(
        index_sum(coefficient, i) * runtime_array("weights", quadrature)[q] * test, q
    )
    permutation = rng.permutation(coefficients.size)
    source = action_kernel(
        expression,
        "staged",
        lambda label, offset: f"w[{offset}]" if label == "coefficients" else f"c[{offset}]",
        a,
        permutation,
    )
    assert source is not None
    # Each private stage is assigned once per entry, after its local reduction.
    assert re.search(r"double sf\d+\[\d+\];", source)
    assert re.search(r"sf\d+\[[^;]+\] = sum\d+;", source)
    assert not re.search(r"sf\d+\[[^;]+\] \+=", source)
    path, library = tmp_path / "kernel.c", tmp_path / "kernel.so"
    path.write_text(source)
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", path, "-o", library], check=True)
    initial = rng.normal(size=coefficients.size)
    actual = initial.copy()
    pointer = ctypes.POINTER(ctypes.c_double)
    ctypes.CDLL(str(library)).staged(
        actual.ctypes.data_as(pointer),
        coefficients.ctypes.data_as(pointer),
        weights.ctypes.data_as(pointer),
        None,
        None,
        None,
        None,
    )
    full = basis[0]
    for values in basis[1:]:
        full = np.kron(full, values)
    expected = initial.copy()
    expected[permutation] += full.T @ (weights.ravel() * (full @ coefficients.ravel()))
    np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


@pytest.mark.skipif(shutil.which("cc") is None, reason="no C compiler")
@pytest.mark.parametrize("direction", range(3))
def test_weighted_broadcast_basis(tmp_path, direction):
    """A constant directional derivative must not hide the other 1D factors."""
    rng = np.random.default_rng(3107)
    q = tuple(index(f"q{k}") for k in range(3))
    a = tuple(index(f"a{k}") for k in range(3))
    basis = [rng.normal(size=(3, 2)) for _ in range(3)]
    basis[direction][:] = [-1, 1]
    weights = [rng.random(3) for _ in range(3)]
    test, weight = num(1), num(1)
    for k in range(3):
        test *= constant(basis[k][0])[a[k]] if k == direction else constant(basis[k])[q[k], a[k]]
        weight *= constant(weights[k])[q[k]]
    test = broadcast_to(test, (3,))[q[direction]]
    values = rng.normal(size=(3, 3, 3))
    expression = index_sum(runtime_array("values", values.shape)[q] * weight * test, q)
    source = action_kernel(
        expression, "weighted", lambda label, offset: f"w[{offset}]", a, range(8)
    )
    assert source is not None
    # There must be no Cartesian-product basis table or premature 3-output stage.
    tables = re.findall(r"static const double \w+((?:\[\d+\])+)", source)
    assert tables and all(len(re.findall(r"\d+", shape)) <= 2 for shape in tables)
    assert max(map(int, re.findall(r"double sf\d+\[(\d+)\]", source))) <= 27
    path, library = tmp_path / "kernel.c", tmp_path / "kernel.so"
    path.write_text(source)
    subprocess.run(["cc", "-O0", "-shared", "-fPIC", path, "-o", library], check=True)
    initial = rng.normal(size=8)
    actual = initial.copy()
    pointer = ctypes.POINTER(ctypes.c_double)
    ctypes.CDLL(str(library)).weighted(
        actual.ctypes.data_as(pointer),
        values.ctypes.data_as(pointer),
        None,
        None,
        None,
        None,
        None,
    )
    full = basis[0] * weights[0][:, None]
    for b, w in zip(basis[1:], weights[1:], strict=True):
        full = np.kron(full, b * w[:, None])
    np.testing.assert_allclose(actual, initial + full.T @ values.ravel(), rtol=1e-12, atol=1e-12)


def test_staged_storage_limit():
    q, i, k = (index(label) for label in ("q", "i", "k"))
    values = runtime_array("values", (200, 200, 2))
    expression = index_sum(index_sum(values[q, i, k], (k,)), (q,))
    assert action_kernel(expression, "large", lambda label, offset: "0", (i,), range(200)) is None


@pytest.mark.parametrize("cell", [basix.CellType.quadrilateral, basix.CellType.hexahedron])
@pytest.mark.parametrize("degree,geometry_degree", [(1, 1), (2, 2)])
@pytest.mark.parametrize("operator", ["mass", "poisson", "helmholtz"])
def test_scalar_actions(tmp_path, ufcx_include, cell, degree, geometry_degree, operator):
    K = basix_cell(cell)
    T = grouped_topology((K,), ((tuple(range(K.num_vertices)),),), ())
    X = basix_space("X", T, basix.ElementFamily.P, geometry_degree)
    D = mesh(
        tensor(tuple(function(X, f"x{k}") for k in range(K.dimension))), EuclideanSpace(K.dimension)
    )
    V = basix_space("V", T, basix.ElementFamily.P, degree)
    u, v = function(V, "u"), function(V, "v")
    U, W = (push_forward(f, D, V.law) for f in (u, v))
    mass = U * W
    gu, gv = gradient(U, D), gradient(W, D)
    stiffness = sum((gu[k] * gv[k] for k in range(K.dimension)), num(0))
    density = {"mass": mass, "poisson": stiffness, "helmholtz": num(3) * mass + num(2) * stiffness}[
        operator
    ]
    f = form(integral(density, D), (v,))
    degree_q = 2 * degree + K.dimension * geometry_degree - 1
    codes = [
        compile_ufcx(f, D, degree_q, coefficients=(u,), sum_factorisation=enabled)
        for enabled in (False, True)
    ]
    assert "double sf0[" in codes[1].source
    functions = [build(code, tmp_path / str(k), ufcx_include)[0] for k, code in enumerate(codes)]
    E = basix.create_element(basix.ElementFamily.P, cell, degree, basix.LagrangeVariant.equispaced)
    G = basix.create_element(
        basix.ElementFamily.P, cell, geometry_degree, basix.LagrangeVariant.equispaced
    )
    rng = np.random.default_rng(991)
    for sign in (1, -1):
        coordinates = np.zeros((G.dim, 3))
        coordinates[:, : K.dimension] = G.points
        coordinates[:, 0] *= sign
        coordinates[:, 0] += 0.08 * G.points[:, 0] * G.points[:, 1]
        if geometry_degree > 1:
            coordinates[:, 1] += 0.05 * G.points[:, 0] ** 2 * G.points[:, 1]
        w, c = rng.normal(size=E.dim), np.empty(0)
        results = [evaluate(w, c, coordinates) for evaluate in functions]
        np.testing.assert_allclose(*results, rtol=1e-10, atol=1e-11)


@pytest.mark.parametrize(
    "cell,kind", [(basix.CellType.triangle, "load"), (basix.CellType.quadrilateral, "mass")]
)
def test_unsupported_forms_keep_existing_emitter(cell, kind):
    f, mesh, coefficients, constants = model(cell, 2, 1, kind)
    ordinary = compile_ufcx(f, mesh, 4, coefficients=coefficients, constants=constants)
    requested = compile_ufcx(
        f, mesh, 4, coefficients=coefficients, constants=constants, sum_factorisation=True
    )
    assert ordinary.source == requested.source


def test_load_with_multiple_coefficients_and_constant(tmp_path, ufcx_include):
    f, mesh, coefficients, constants = model(basix.CellType.quadrilateral, 2, 2, "load")
    codes = [
        compile_ufcx(
            f, mesh, 5, coefficients=coefficients, constants=constants, sum_factorisation=enabled
        )
        for enabled in (False, True)
    ]
    functions = [build(code, tmp_path / str(k), ufcx_include)[0] for k, code in enumerate(codes)]
    E = basix.create_element(
        basix.ElementFamily.P, basix.CellType.quadrilateral, 2, basix.LagrangeVariant.equispaced
    )
    coordinates = np.zeros((E.dim, 3))
    coordinates[:, :2] = E.points
    coordinates[:, 0] += 0.05 * E.points[:, 0] * E.points[:, 1]
    w = np.random.default_rng(997).normal(size=2 * E.dim)
    results = [evaluate(w, np.array([0.3]), coordinates) for evaluate in functions]
    np.testing.assert_allclose(*results, rtol=1e-11, atol=1e-12)


def test_storage_fallback_rebuilds_ordinary_expression(monkeypatch):
    import uflx_compiler.ufcx as ufcx

    f, mesh, coefficients, constants = model(basix.CellType.quadrilateral, 2, 2, "load")
    ordinary = compile_ufcx(f, mesh, 5, coefficients=coefficients, constants=constants)
    monkeypatch.setattr(ufcx, "action_kernel", lambda *args: None)
    requested = compile_ufcx(
        f, mesh, 5, coefficients=coefficients, constants=constants, sum_factorisation=True
    )
    assert ordinary.source == requested.source


def test_zero_action():
    f, mesh, _, _ = model(basix.CellType.quadrilateral, 2, 2, "load")
    zero = form(integral(num(0), mesh), f.arguments)
    ordinary = compile_ufcx(zero, mesh, 5)
    requested = compile_ufcx(zero, mesh, 5, sum_factorisation=True)
    assert ordinary.source == requested.source
