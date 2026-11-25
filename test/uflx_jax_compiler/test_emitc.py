"""Compile and run generated C kernels."""

import ctypes
import shutil
import subprocess

import basix
import jax
import numpy as np
import pytest

import uflx_jax_compiler
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space, push_forward
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import local_space
from uflx_jax_compiler.emitc import _require_mlir_tools


@pytest.fixture(scope="module", autouse=True)
def require_emitc_tools():
    if not all(shutil.which(tool) for tool in ("clang-format", "cc")):
        pytest.skip("requires clang-format and a C compiler")
    try:
        _require_mlir_tools()
    except (FileNotFoundError, RuntimeError) as error:
        pytest.skip(str(error))


def compile_kernel(source, name, tmp_path):
    library = tmp_path / f"{name}.so"
    subprocess.run(
        ["cc", "-std=c99", "-pedantic-errors", "-shared", "-fPIC", "-x", "c", "-", "-o", library],
        input=source,
        text=True,
        check=True,
    )
    return getattr(ctypes.CDLL(str(library)), name)


def test_p1_tetrahedron(tmp_path):
    cell = basix.CellType.tetrahedron
    T = topology(cell)
    p1 = basix.create_element(basix.ElementFamily.P, cell, 1)
    V = local_space(T, p1)
    x = function(local_space(T, p1, shape=(3,)), "x")
    mesh = create_mesh(x, euclidean_space(3))
    u, v = function(V, "u"), function(V, "v")
    gu, gv = (gradient(push_forward(f, mesh, V.law), mesh) for f in (u, v))
    a = lower(integral(gv @ gu, mesh))
    form_a = form(apply_quadrature(a, *basix.make_quadrature(cell, 0)), (v, u))
    kernel = jax.jit(uflx_jax_compiler.trace(form_a, x))

    vertices = np.array([[0.2, 0, 0.3], [1.3, 0.1, 0], [0, 1.2, 0.1], [0.1, 0.2, 1.1]])
    source = uflx_jax_compiler.emit_c(jax.export.export(kernel)(vertices), "stiffness", tmp_path)
    stiffness = compile_kernel(source, "stiffness", tmp_path)
    stiffness.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, shape=shape, flags="C_CONTIGUOUS")
        for shape in ((4, 3), (4, 4))
    ]
    stiffness.restype = None
    result = np.empty((4, 4))
    stiffness(vertices, result)
    np.testing.assert_allclose(result, kernel(vertices), atol=1e-13)


def test_scalar_reduction(tmp_path):
    values = np.array([1.0, -2.0, 3.0])
    kernel = jax.jit(lambda x: (x * x).sum())
    source = uflx_jax_compiler.emit_c(jax.export.export(kernel)(values), "energy", tmp_path)
    assert (tmp_path / "energy.c").read_text() == source
    energy = compile_kernel(source, "energy", tmp_path)
    energy.argtypes = [np.ctypeslib.ndpointer(dtype=np.float64)] * 2
    energy.restype = None
    result = np.array(np.nan)
    energy(values, result)
    np.testing.assert_allclose(result, kernel(values))
