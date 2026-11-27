"""Compare both real cell-kernel ABIs with DOLFINx's native Poisson assembly."""

import basix
import jax
import numpy as np
import pytest

pytest.importorskip("dolfinx")
import ufl
from dolfinx import fem, mesh
from mpi4py import MPI

import uflx_dolfinx
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.passes import lower
from uflx_basix.quadrature import apply_quadrature


@pytest.mark.parametrize("dtype", [np.float32, np.float64])
@pytest.mark.parametrize("x64_enabled", [False, True])
def test_poisson_assembly_precision(dtype, x64_enabled):
    square = mesh.create_unit_square(MPI.COMM_SELF, 2, 2, dtype=dtype)
    P1 = fem.functionspace(square, ("Lagrange", 1))
    trial_space = P1.clone()
    u = function(trial_space, "u")
    v = function(P1, "v")
    k = uflx_dolfinx.Function(P1, name="k")
    k.interpolate(lambda x: 1 + x[0])
    assert uflx_dolfinx.Function(P1, name="k") is k
    assert uflx_dolfinx.Function(P1, name="other").data is not k.data
    c = uflx_dolfinx.Constant(square, dtype(2), name="c")
    domain = v.domain
    a = integral(c * k * (gradient(v, domain) @ gradient(u, domain)), domain)
    rule = basix.make_quadrature(square.basix_cell(), degree=1)
    a = form(apply_quadrature(lower(a), *rule), (v, u))
    with jax.enable_x64(x64_enabled):
        if (dtype == np.float64) != x64_enabled:
            with pytest.raises(ValueError, match="JAX_ENABLE_X64"):
                uflx_dolfinx.form(a)
            return
        other_dtype = np.float32 if dtype == np.float64 else np.float64
        with pytest.raises(ValueError, match=f"constant has dtype .*expected {np.dtype(dtype)}"):
            uflx_dolfinx.Constant(square, other_dtype(2), name="c")
        bound = uflx_dolfinx.form(a)
        assert bound.function_spaces[0] is P1
        assert bound.function_spaces[1] is trial_space
        A = fem.assemble_matrix(bound)
    native = (
        c.data
        * k.data
        * ufl.inner(ufl.grad(ufl.TrialFunction(trial_space)), ufl.grad(ufl.TestFunction(P1)))
        * ufl.dx
    )
    expected = fem.assemble_matrix(fem.form(native, dtype=dtype))
    assert A.data.dtype == dtype
    tol = 20 * np.finfo(dtype).eps
    np.testing.assert_allclose(A.data, expected.data, rtol=tol, atol=tol)
    other = uflx_dolfinx.Constant(square, dtype(2), name="other")
    c.value[...] = 4
    assert uflx_dolfinx.Constant(square, dtype(2), name="c") is c
    assert c.value == 4 and other.value == 2
    np.testing.assert_allclose(fem.assemble_matrix(bound).data, 2 * A.data, rtol=tol, atol=tol)


def test_space_from_native_elements(monkeypatch):
    square = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    other = mesh.create_unit_square(MPI.COMM_SELF, 2, 2)
    assert square.name == other.name
    other_space = fem.functionspace(other, ("Lagrange", 1))
    spaces = [
        fem.functionspace(square, spec)
        for spec in [("Lagrange", 1), ("Lagrange", 1, (2,)), ("N1curl", 1)]
    ]

    c = uflx_dolfinx.Constant(square, np.float64(1), name="c")
    assert c is not uflx_dolfinx.Constant(other, np.float64(1), name="c")

    def unexpected(*args):
        pytest.fail("space conversion consulted UFL metadata")

    monkeypatch.setattr(type(square), "ufl_domain", unexpected)
    monkeypatch.setattr(type(spaces[0]), "ufl_element", unexpected)
    scalar, vector, edge = map(uflx_dolfinx.functionspace, spaces)
    assert scalar.domain is vector.domain is edge.domain
    assert scalar.domain is not uflx_dolfinx.functionspace(other_space).domain
    square.name = "renamed"
    assert scalar is uflx_dolfinx.functionspace(spaces[0])
    assert c is uflx_dolfinx.Constant(square, np.float64(1), name="c")
    assert scalar.value_shape == ()
    assert vector.value_shape == edge.value_shape == (2,)
    assert vector.reference_space.num_dofs == 6
    assert edge.reference_space.num_dofs == 3
