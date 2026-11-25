"""Native C emission for an affine tetrahedral stiffness matrix."""

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


@pytest.mark.skipif(
    not all(shutil.which(tool) for tool in ("mlir-opt", "mlir-translate", "clang-format", "cc")),
    reason="requires MLIR tools, clang-format, and a C compiler",
)
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
    source = uflx_jax_compiler.emit_c(jax.export.export(kernel)(vertices), name="stiffness")
    library = tmp_path / "stiffness.so"
    subprocess.run(
        ["cc", "-std=c99", "-pedantic-errors", "-shared", "-fPIC", "-x", "c", "-", "-o", library],
        input=source,
        text=True,
        check=True,
    )
    stiffness = ctypes.CDLL(str(library)).stiffness
    stiffness.argtypes = [
        np.ctypeslib.ndpointer(dtype=np.float64, shape=shape, flags="C_CONTIGUOUS")
        for shape in ((4, 3), (4, 4))
    ]
    stiffness.restype = None
    result = np.empty((4, 4))
    stiffness(vertices, result)
    np.testing.assert_allclose(result, kernel(vertices), atol=1e-13)
