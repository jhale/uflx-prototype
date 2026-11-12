"""Blocked Basix spaces and native UFCx vector/tensor kernels."""

import ctypes
from pathlib import Path

import basix
import numpy as np
import pytest
from test_basix import quadrilaterals, square
from test_ufcx import boundary_mesh, build, interior_mesh, model, paired_geometry

from uflx_compiler import compile_form, compile_ufcx, evaluate
from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    form,
    function,
    gradient,
    integral,
    num,
    pull_back,
    push_forward,
    side_map,
)
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.spaces import space


def elasticity(cell, degree, geometry_degree):
    """Native vector elasticity with two runtime Lame constants."""
    _, D, _, _ = model(cell, degree, geometry_degree, "mass")
    d = D.dimension
    V = basix_space("displacement", D.topology, basix.ElementFamily.P, degree, shape=(d,))
    u, v = function(V, "u"), function(V, "v")
    U, W = (push_forward(f, D, V.law) for f in (u, v))
    du, dv = gradient(U, D), gradient(W, D)
    lam, mu = (function(space("material", scalar_type="real"), name) for name in ("lambda", "mu"))
    divu = sum((du[k, k] for k in range(d)), num(0))
    divv = sum((dv[k, k] for k in range(d)), num(0))
    strain = sum(((du[i, j] + du[j, i]) * dv[i, j] for i in range(d) for j in range(d)), num(0))
    return form(integral(lam * divu * divv + mu * strain, D), (v, u)), D, (lam, mu)


def reference_space(cell, degree, geometry_degree, shape):
    import basix.ufl
    import ufl

    d = len(basix.cell.topology(cell)) - 1
    coord = basix.ufl.element(
        "P", cell, geometry_degree, basix.LagrangeVariant.equispaced, shape=(d,)
    )
    D = ufl.Mesh(coord)
    V = ufl.FunctionSpace(
        D, basix.ufl.element("P", cell, degree, basix.LagrangeVariant.equispaced, shape=shape)
    )
    return D, V


def geometry(cell, degree):
    E = basix.create_element(basix.ElementFamily.P, cell, degree, basix.LagrangeVariant.equispaced)
    d = E.points.shape[1]
    xyz = np.zeros((E.dim, 3))
    xyz[:, :d] = 1.2 * E.points + 0.1
    xyz[:, 0] += 0.15 * E.points[:, 1]
    if degree > 1:
        xyz[:, 0] += 0.09 * E.points[:, 0] * E.points[:, 1]
        if d == 3:
            xyz[:, 2] += 0.06 * E.points[:, 0] * E.points[:, 2]
    return xyz


def check_kernel(code, reference, folder, xyz, w, c, facets=None, permutations=None):
    jit = pytest.importorskip("ffcx.codegeneration.jit")
    evaluate_kernel, lib = build(code, folder, Path(jit.__file__).parent)
    forms, module, _ = jit.compile_forms(
        [reference], options={"scalar_type": "float64"}, cache_dir=folder / "ffcx"
    )
    ffi = module.ffi
    expected = np.zeros(code.tensor_shape or ())
    forms[0].form_integrals[0].tabulate_tensor_float64(
        *(ffi.cast("double*", a.ctypes.data) for a in (expected, w, c, xyz)),
        ffi.NULL if facets is None else ffi.new("int[]", facets),
        ffi.NULL if permutations is None else ffi.new("uint8_t[]", permutations),
        ffi.NULL,
    )
    actual = evaluate_kernel(w, c, xyz, facets, permutations)
    np.testing.assert_allclose(actual, expected, rtol=1e-10, atol=1e-11)
    descriptor = ffi.cast("ufcx_form*", ctypes.addressof(ctypes.c_char.in_dll(lib, code.name)))
    for i in range(len(code.tensor_shape) + len(code.coefficients)):
        assert descriptor.finite_element_hashes[i] == forms[0].finite_element_hashes[i]
    return actual


@pytest.mark.parametrize(
    "cell,degree,geometry_degree",
    [(basix.CellType.triangle, p, g) for p in (1, 2, 3) for g in (1, 2)]
    + [
        (K, p, g)
        for K, p in (
            (basix.CellType.quadrilateral, 2),
            (basix.CellType.tetrahedron, 2),
            (basix.CellType.hexahedron, 1),
        )
        for g in (1, 2)
    ],
)
def test_native_elasticity(tmp_path, cell, degree, geometry_degree):
    ufl = pytest.importorskip("ufl")
    f, D, constants = elasticity(cell, degree, geometry_degree)
    qdegree = 2 * degree + 2
    code = compile_ufcx(f, D, qdegree, constants=constants)
    mesh, V = reference_space(cell, degree, geometry_degree, (D.dimension,))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    lam, mu = ufl.Constant(mesh), ufl.Constant(mesh)
    reference = (
        lam * ufl.div(u) * ufl.div(v)
        + 2 * mu * ufl.inner(ufl.sym(ufl.grad(u)), ufl.sym(ufl.grad(v)))
    ) * ufl.dx(metadata={"quadrature_degree": qdegree})
    n = (
        D.dimension
        * basix.create_element(
            basix.ElementFamily.P, cell, degree, basix.LagrangeVariant.equispaced
        ).dim
    )
    assert code.tensor_shape == (n, n)
    matrix = check_kernel(
        code,
        reference,
        tmp_path,
        geometry(cell, geometry_degree),
        np.empty(0),
        np.array([1.3, 0.8]),
    )
    np.testing.assert_allclose(matrix, matrix.T, rtol=1e-11, atol=1e-11)
    translations = np.tile(np.eye(D.dimension), (n // D.dimension, 1))
    np.testing.assert_allclose(matrix @ translations, 0, atol=1e-10)


@pytest.mark.parametrize("kind", ["cell", "exterior", "interior"])
@pytest.mark.parametrize("cell", [basix.CellType.triangle, basix.CellType.hexahedron])
def test_blocked_coefficient_packing_on_cells_and_facets(tmp_path, kind, cell):
    ufl = pytest.importorskip("ufl")
    _, D, _, _ = model(cell, 1, 2, "mass")
    d = D.dimension
    V = basix_space("vectors", D.topology, basix.ElementFamily.P, 1, shape=(d,))
    u, v, a = (function(V, label) for label in ("u", "v", "a"))
    U, W, A = (push_forward(x, D, V.law) for x in (u, v, a))
    _, VV = reference_space(cell, 1, 2, (d,))
    uu, vv, aa = ufl.TrialFunction(VV), ufl.TestFunction(VV), ufl.Coefficient(VV)
    xyz = geometry(cell, 2)
    facets = permutations = None
    width = 1
    if kind == "cell":
        domain, measure = D, ufl.dx
    elif kind == "exterior":
        domain, measure = boundary_mesh(D), ufl.ds
        trace = side_map(D, domain, "inside")
        U, W, A = (pull_back(x, trace) for x in (U, W, A))
        facets = [1]
    else:
        domain, measure = interior_mesh(D), ufl.dS
        plus, minus = side_map(D, domain, "+"), side_map(D, domain, "-")
        U, W = (pull_back(x, plus) - pull_back(x, minus) for x in (U, W))
        A = num(0.5) * (pull_back(A, plus) + pull_back(A, minus))
        uu, vv, aa = ufl.jump(uu), ufl.jump(vv), ufl.avg(aa)
        width = 2
        facets, permutations = [0, 1], [1, 0]
        X = basix.create_element(basix.ElementFamily.P, cell, 2, basix.LagrangeVariant.equispaced)
        xyz = paired_geometry(cell, X, facets, permutations)
    value = sum((A[i] * U[j] * W[i] for i in range(d) for j in range(d)), num(0))
    reference = sum(aa[i] * uu[j] * vv[i] for i in range(d) for j in range(d))
    code = compile_ufcx(form(integral(value, domain), (v, u)), D, 4, coefficients=(a,))
    n = width * d * V.element(0).dim
    assert code.tensor_shape == (n, n)
    assert code.coefficient_offsets == (0,)
    assert code.coefficients == (a,)
    check_kernel(
        code,
        reference * measure(metadata={"quadrature_degree": 4}),
        tmp_path,
        xyz,
        np.random.default_rng(3).normal(size=n),
        np.empty(0),
        facets,
        permutations,
    )


@pytest.mark.parametrize("cell", [basix.CellType.triangle, basix.CellType.quadrilateral])
@pytest.mark.parametrize("shape", [(2,), (2, 2)])
def test_blocked_fields_in_array_compiler_and_oracle(cell, shape):
    D = square(basix_cell(cell))[0] if cell == basix.CellType.triangle else quadrilaterals()[0]
    V = basix_space("blocked", D.topology, basix.ElementFamily.P, 2, shape=shape)
    scalar = basix_space("scalar", D.topology, basix.ElementFamily.P, 2)
    assert V.num_dofs == scalar.num_dofs * np.prod(shape)
    for row, base in zip(V.dofmap, scalar.dofmap, strict=True):
        assert row == tuple(V.block_size * i + c for i in base for c in range(V.block_size))
    u = function(V, "u")
    U = push_forward(u, D, V.law)
    f = integral(sum((U[idx] * U[idx] for idx in np.ndindex(shape)), num(0)), D)
    env = {"coefficients:u": np.random.default_rng(7).normal(size=V.num_dofs)}
    np.testing.assert_allclose(evaluate(compile_form(f, 6), env), oracle(f, env)(()), rtol=1e-8)


def test_invalid_block_shapes():
    _, D, _, _ = model(basix.CellType.triangle, 1, 1, "mass")
    with pytest.raises(ValueError, match="positive"):
        basix_space("bad", D.topology, basix.ElementFamily.P, 1, shape=(0,))
    with pytest.raises(ValueError, match="scalar base"):
        basix_space("bad", D.topology, basix.ElementFamily.RT, 1, shape=(2,))


@pytest.mark.parametrize("rank", [0, 1, 2])
def test_tensor_coefficients_and_rectangular_arguments(tmp_path, rank):
    ufl = pytest.importorskip("ufl")
    cell = basix.CellType.triangle
    _, D, _, _ = model(cell, 1, 1, "mass")
    scalar = basix_space("scalar", D.topology, basix.ElementFamily.P, 1)
    vector = basix_space("vector", D.topology, basix.ElementFamily.P, 1, shape=(2,))
    tensor = basix_space("tensor", D.topology, basix.ElementFamily.P, 1, shape=(2, 2))
    a, b, u, v = (
        function(V, label)
        for V, label in ((scalar, "a"), (tensor, "b"), (scalar, "u"), (vector, "v"))
    )
    A, B, U, W = (push_forward(f, D, f.space.law) for f in (a, b, u, v))
    mesh, VV = reference_space(cell, 1, 1, (2,))
    SS = ufl.FunctionSpace(mesh, basix.ufl.element("P", cell, 1, basix.LagrangeVariant.equispaced))
    TT = ufl.FunctionSpace(
        mesh, basix.ufl.element("P", cell, 1, basix.LagrangeVariant.equispaced, shape=(2, 2))
    )
    aa, bb = ufl.Coefficient(SS), ufl.Coefficient(TT)
    uu, vv = ufl.TrialFunction(SS), ufl.TestFunction(VV)
    value = A * (B[0, 1] + num(2) * B[1, 0])
    reference = aa * (bb[0, 1] + 2 * bb[1, 0])
    if rank >= 1:
        value = value * (W[0] + num(3) * W[1])
        reference = reference * (vv[0] + 3 * vv[1])
    if rank == 2:
        value = value * U
        reference = reference * uu
    arguments = (v, u)[:rank]
    code = compile_ufcx(form(integral(value, D), arguments), D, 4, coefficients=(a, b))
    assert code.coefficient_offsets == (0, 3)
    assert code.tensor_shape == (6, 3)[:rank]
    check_kernel(
        code,
        reference * ufl.dx(metadata={"quadrature_degree": 4}),
        tmp_path,
        geometry(cell, 1),
        np.arange(15, dtype=float) / 7,
        np.empty(0),
    )
