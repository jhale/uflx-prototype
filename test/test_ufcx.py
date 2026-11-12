"""Compile against the real UFCx header and compare local tensors with FFCx."""

import ctypes
import os
import shutil
import subprocess
from pathlib import Path

import basix
import numpy as np
import pytest
from testutil import grouped_topology

from uflx_compiler import compile_ufcx
from uflx_spec_interpreter import (
    RD,
    form,
    function,
    gradient,
    image,
    integral,
    num,
    push_forward,
    tensor,
)
from uflx_toy.basix_cells import basix_cell
from uflx_toy.basix_spaces import basix_space
from uflx_toy.spaces import space


@pytest.fixture
def ufcx_include():
    if shutil.which("cc") is None:
        pytest.skip("no C compiler")
    if path := os.environ.get("UFCX_INCLUDE_DIR"):
        return Path(path)
    ffcx = pytest.importorskip("ffcx")
    return Path(ffcx.__file__).parent / "codegeneration"


def model(cell, degree, coordinate_degree, kind, cells=1):
    K = basix_cell(cell)
    vertices = tuple(range(K.num_vertices))
    T = grouped_topology(
        (K,),
        (tuple(tuple(v + slot * K.num_vertices for v in vertices) for slot in range(cells)),),
        (),
    )
    X = basix_space("X", T, basix.ElementFamily.P, coordinate_degree)
    coords = tuple(function(X, f"x{i}") for i in range(K.dimension))
    D = image(tensor(coords), RD(K.dimension))
    V = basix_space("V", T, basix.ElementFamily.P, degree)
    u, v, a, b = (function(V, s) for s in ("u", "v", "a", "b"))
    U, W, A, B = (push_forward(f, D, V.law) for f in (u, v, a, b))
    c = function(space("constant", scalar_type="real"), "c")
    weight = A + num(2) * B + c
    match kind:
        case "mass":
            f = form(integral(weight * W * U, D), (v, u))
        case "stiffness":
            gu, gv = gradient(U, D), gradient(W, D)
            product = sum((gu[i] * gv[i] for i in range(K.dimension)), num(0))
            f = form(integral(weight * product, D), (v, u))
        case "load":
            f = form(integral(weight * W, D), (v,))
        case "energy":
            f = integral(weight * weight, D)
    return f, D, (a, b), (c,)


def build(code, folder, include):
    folder.mkdir(exist_ok=True)
    source, library = folder / "kernel.c", folder / "kernel.so"
    # This declaration checks the generated function against UFCx's actual typedef.
    source.write_text(
        code.source + f"\nufcx_tabulate_tensor_float64* check = {code.name}_kernel;\n"
    )
    subprocess.run(
        [
            "cc",
            "-std=c99",
            "-pedantic-errors",
            "-Werror=incompatible-pointer-types",
            "-O2",
            "-shared",
            "-fPIC",
            "-I" + str(include),
            str(source),
            "-o",
            str(library),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    lib = ctypes.CDLL(str(library))
    kernel = getattr(lib, code.name + "_kernel")
    pointer = ctypes.POINTER(ctypes.c_double)

    def evaluate(w, c, coordinates, facet=None, permutations=None):
        A = np.zeros(code.tensor_shape or ())
        facets = [] if facet is None else ([facet] if isinstance(facet, int) else facet)
        entity = (ctypes.c_int * len(facets))(*facets) if facets else None
        permutation = (
            (ctypes.c_uint8 * len(permutations))(*permutations)
            if permutations is not None
            else None
        )
        kernel(
            *(v.ctypes.data_as(pointer) for v in (A, w, c, coordinates)),
            entity,
            permutation,
            None,
        )
        return A

    return evaluate, lib


def reference(cell, degree, coordinate_degree, kind):
    ufl = pytest.importorskip("ufl")
    pytest.importorskip("ffcx")
    import basix.ufl

    coord = basix.ufl.element(
        "Lagrange",
        cell,
        coordinate_degree,
        basix.LagrangeVariant.equispaced,
        shape=(len(basix.cell.topology(cell)) - 1,),
    )
    D = ufl.Mesh(coord)
    V = ufl.FunctionSpace(
        D, basix.ufl.element("Lagrange", cell, degree, basix.LagrangeVariant.equispaced)
    )
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a, b, c = ufl.Coefficient(V), ufl.Coefficient(V), ufl.Constant(D)
    weight = a + 2 * b + c
    expressions = {
        "mass": weight * v * u,
        "stiffness": weight * ufl.inner(ufl.grad(u), ufl.grad(v)),
        "load": weight * v,
        "energy": weight**2,
    }
    return expressions[kind] * ufl.dx(metadata={"quadrature_degree": 6}), coord.basix_hash()


@pytest.mark.parametrize(
    "cell,degree,coordinate_degree",
    [
        (basix.CellType.triangle, 1, 1),
        (basix.CellType.triangle, 3, 2),
        (basix.CellType.quadrilateral, 2, 1),
        (basix.CellType.tetrahedron, 1, 1),
    ],
)
@pytest.mark.parametrize("kind", ["energy", "load", "mass", "stiffness"])
def test_local_tensors_match_ffcx(tmp_path, ufcx_include, cell, degree, coordinate_degree, kind):
    ffcx_jit = pytest.importorskip("ffcx.codegeneration.jit")
    f, mesh, coefficients, constants = model(cell, degree, coordinate_degree, kind)
    code = compile_ufcx(f, mesh, 6, coefficients=coefficients, constants=constants)
    evaluate, lib = build(code, tmp_path, ufcx_include)
    uf, coordinate_hash = reference(cell, degree, coordinate_degree, kind)
    forms, module, _ = ffcx_jit.compile_forms(
        [uf],
        options={"scalar_type": "float64"},
        cache_dir=tmp_path / "ffcx",
    )
    kernel = forms[0].form_integrals[0].tabulate_tensor_float64
    ffi = module.ffi
    E = basix.create_element(basix.ElementFamily.P, cell, degree, basix.LagrangeVariant.equispaced)
    X = basix.create_element(
        basix.ElementFamily.P, cell, coordinate_degree, basix.LagrangeVariant.equispaced
    )
    rng = np.random.default_rng(13)
    assert code.coefficient_offsets == (0, E.dim)
    for scale in (0.7, 1.3):
        xyz = np.zeros((X.dim, 3))
        dim = X.points.shape[1]
        xyz[:, :dim] = scale * X.points + 0.1
        if coordinate_degree > 1:
            xyz[:, 0] += 0.07 * X.points[:, 0] * X.points[:, 1]
        w, c = rng.normal(size=2 * E.dim), np.array([0.3])
        expected = np.zeros(code.tensor_shape or ())
        kernel(
            *(ffi.cast("double*", v.ctypes.data) for v in (expected, w, c, xyz)),
            ffi.NULL,
            ffi.NULL,
            ffi.NULL,
        )
        assert np.allclose(evaluate(w, c, xyz), expected, rtol=1e-11, atol=1e-12)
    # Read descriptors using FFCx's CFFI declarations from the compiled library.
    address = ctypes.addressof(ctypes.c_char.in_dll(lib, code.name))
    descriptor = ffi.cast("ufcx_form*", address)
    assert descriptor.rank == len(code.tensor_shape)
    assert descriptor.num_coefficients == 2 and descriptor.num_constants == 1
    assert [descriptor.form_integral_offsets[i] for i in range(6)] == [0, 1, 1, 1, 1, 1]
    assert descriptor.form_integral_ids[0] == -1
    integral = descriptor.form_integrals[0]
    assert integral.domain == cell.value
    assert integral.coordinate_element_hash == coordinate_hash
    assert integral.enabled_coefficients[0] and integral.enabled_coefficients[1]
    assert not integral.needs_facet_permutations
    assert integral.tabulate_tensor_float32 == ffi.NULL
    for i in range(len(code.tensor_shape) + 2):
        assert descriptor.finite_element_hashes[i] == E.hash()


def test_cell_kernels_reject_multicell_configurations():
    # The adapter must not silently drop all but the first local cell.
    f, mesh, coefficients, constants = model(basix.CellType.triangle, 3, 2, "mass", cells=9)
    with pytest.raises(NotImplementedError, match="exactly one parent cell slot"):
        compile_ufcx(f, mesh, 4, coefficients=coefficients, constants=constants)


def simple_form(zero=False, discontinuous=False, cells=1):
    _, mesh, _, _ = model(basix.CellType.triangle, 1, 1, "mass", cells=cells)
    V = basix_space("V", mesh.topology, basix.ElementFamily.P, 1, discontinuous)
    u, v = function(V, "u"), function(V, "v")
    U, W = (push_forward(f, mesh, V.law) for f in (u, v))
    return form(integral((num(0) if zero else U) * W, mesh), (v, u)), mesh


@pytest.mark.parametrize("zero", [False, True])
def test_no_inputs_and_zero_forms(tmp_path, ufcx_include, zero):
    f, mesh = simple_form(zero)
    code = compile_ufcx(f, mesh, 2)
    evaluate, _ = build(code, tmp_path, ufcx_include)
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    expected = np.zeros((3, 3)) if zero else (np.ones((3, 3)) + np.eye(3)) / 24
    assert code.coefficients == code.constants == ()
    assert np.allclose(evaluate(np.empty(0), np.empty(0), xyz), expected)


def test_unsupported_integrals_and_input_lists():
    from uflx_spec_interpreter import replace

    f, mesh, fields, constants = model(basix.CellType.triangle, 1, 1, "mass")
    with pytest.raises(ValueError, match="exactly once"):
        compile_ufcx(f, mesh, 2, coefficients=fields[:1])
    with pytest.raises(ValueError, match="C identifier"):
        compile_ufcx(f, mesh, 2, name="bad-name")
    z = function(space("complex", scalar_type="complex"), "z")
    from uflx_spec_interpreter import real_part

    with pytest.raises(NotImplementedError, match="real-valued inputs"):
        compile_ufcx(replace(f, {constants[0]: real_part(z)}), mesh, 2)


@pytest.mark.parametrize("integral_type", ["cell", "exterior_facet", "interior_facet"])
def test_dolfinx_assembles_generated_form(tmp_path, ufcx_include, integral_type):
    dolfinx = pytest.importorskip("dolfinx")
    MPI = pytest.importorskip("mpi4py.MPI")
    ufl = pytest.importorskip("ufl")
    import basix.ufl
    from dolfinx.fem.forms import form_cpp_class

    f, mesh = simple_form(
        discontinuous=integral_type == "interior_facet",
        cells=2 if integral_type == "interior_facet" else 1,
    )
    if integral_type == "exterior_facet":
        from testutil import side_map

        from uflx_spec_interpreter import pull_back

        boundary = boundary_mesh(mesh)
        trace = pull_back(f.expression.expression, side_map(mesh, boundary, "inside"))
        f = form(integral(trace, boundary), f.arguments)
    elif integral_type == "interior_facet":
        from testutil import side_map

        from uflx_spec_interpreter import pull_back

        boundary = interior_mesh(mesh)
        plus, minus = side_map(mesh, boundary, "+"), side_map(mesh, boundary, "-")
        traces = []
        for argument in f.arguments:
            value = push_forward(argument, mesh, argument.space.law)
            traces.append(pull_back(value, plus) - pull_back(value, minus))
        f = form(integral(traces[0] * traces[1], boundary), f.arguments)
    code = compile_ufcx(f, mesh, 2)
    _, lib = build(code, tmp_path, ufcx_include)
    coordinate_element = basix.ufl.element(
        "Lagrange", "triangle", 1, basix.LagrangeVariant.equispaced, shape=(2,)
    )
    cells = np.array([[0, 1, 2], [1, 3, 2]], dtype=np.int64)
    points = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]])
    domain = dolfinx.mesh.create_mesh(MPI.COMM_SELF, cells, ufl.Mesh(coordinate_element), points)
    V = dolfinx.fem.functionspace(
        domain,
        basix.ufl.element(
            "Lagrange",
            "triangle",
            1,
            basix.LagrangeVariant.equispaced,
            discontinuous=integral_type == "interior_facet",
        ),
    )
    address = ctypes.addressof(ctypes.c_char.in_dll(lib, code.name))
    measure = {"cell": ufl.dx, "exterior_facet": ufl.ds, "interior_facet": ufl.dS}[integral_type]
    itype = getattr(dolfinx.fem.IntegralType, integral_type)
    if integral_type != "cell":
        domain.topology.create_connectivity(domain.topology.dim - 1, domain.topology.dim)
    cpp_form = form_cpp_class(np.float64)(
        [address],
        [V._cpp_object, V._cpp_object],
        [],
        [],
        {itype: []},
        [],
        domain._cpp_object,
    )
    generated = dolfinx.fem.Form(cpp_form, domain, [V, V])
    trial, test = ufl.TrialFunction(V), ufl.TestFunction(V)
    expected = dolfinx.fem.form(
        (ufl.jump(trial) * ufl.jump(test) if integral_type == "interior_facet" else trial * test)
        * measure
    )
    actual_matrix = dolfinx.fem.assemble_matrix(generated)
    expected_matrix = dolfinx.fem.assemble_matrix(expected)
    actual_matrix.scatter_reverse()
    expected_matrix.scatter_reverse()
    assert np.allclose(actual_matrix.to_dense(), expected_matrix.to_dense())


@pytest.mark.parametrize("kind", ["mass", "divergence"])
def test_piola_elements_match_ffcx(tmp_path, ufcx_include, kind):
    ufl = pytest.importorskip("ufl")
    jit = pytest.importorskip("ffcx.codegeneration.jit")
    import basix.ufl

    from uflx_spec_interpreter import divergence

    _, D = simple_form()
    V = basix_space("RT", D.topology, basix.ElementFamily.RT, 1)
    u, v = function(V, "u"), function(V, "v")
    U, W = (push_forward(f, D, V.law) for f in (u, v))
    integrand = U[0] * W[0] + U[1] * W[1] if kind == "mass" else divergence(U, D) * divergence(W, D)
    code = compile_ufcx(form(integral(integrand, D), (v, u)), D, 4)
    evaluate, _ = build(code, tmp_path, ufcx_include)
    coordinate = basix.ufl.element("P", "triangle", 1, basix.LagrangeVariant.equispaced, shape=(2,))
    domain = ufl.Mesh(coordinate)
    space = ufl.FunctionSpace(domain, basix.ufl.element("RT", "triangle", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    expr = ufl.inner(trial, test) if kind == "mass" else ufl.div(trial) * ufl.div(test)
    forms, module, _ = jit.compile_forms(
        [expr * ufl.dx(metadata={"quadrature_degree": 4})],
        options={"scalar_type": "float64"},
        cache_dir=tmp_path / "ffcx",
    )
    xyz = np.array([[0.1, 0.2, 0.0], [1.3, 0.1, 0.0], [0.2, 0.8, 0.0]])
    expected = np.zeros(code.tensor_shape)
    ffi = module.ffi
    forms[0].form_integrals[0].tabulate_tensor_float64(
        ffi.cast("double*", expected.ctypes.data),
        ffi.NULL,
        ffi.NULL,
        ffi.cast("double*", xyz.ctypes.data),
        ffi.NULL,
        ffi.NULL,
        ffi.NULL,
    )
    assert np.allclose(evaluate(np.empty(0), np.empty(0), xyz), expected, atol=1e-12)


def test_tensor_constants_and_vertex_geometry(tmp_path, ufcx_include):
    from testutil import vertex_mesh

    K = basix_cell(basix.CellType.triangle)
    T = grouped_topology((K,), (((0, 1, 2),),), ())
    D = vertex_mesh(T, [(0, 0), (1, 0), (0, 1)])
    a = function(space("a", (2,), scalar_type="real"), "a")
    b = function(space("b", scalar_type="real"), "b")
    code = compile_ufcx(integral(a[0] + num(2) * a[1] + b, D), D, 2, constants=(a, b))
    assert code.constant_offsets == (0, 2)
    evaluate, _ = build(code, tmp_path, ufcx_include)
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    assert np.isclose(evaluate(np.empty(0), np.array([1.0, 3.0, 4.0]), xyz), 5.5)


def test_rectangular_tensor_argument_order(tmp_path, ufcx_include):
    _, D = simple_form()
    P = basix.ElementFamily.P
    V = basix_space("rows", D.topology, P, 1)
    U = basix_space("columns", D.topology, P, 2)
    v, u = function(V, "v"), function(U, "u")
    integrand = push_forward(v, D, V.law) * push_forward(u, D, U.law)
    code = compile_ufcx(form(integral(integrand, D), (v, u)), D, 4)
    evaluate, _ = build(code, tmp_path, ufcx_include)
    assert code.tensor_shape == (3, 6)
    xyz = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    points, weights = basix.make_quadrature(basix.CellType.triangle, 4)
    row = basix.create_element(P, basix.CellType.triangle, 1, basix.LagrangeVariant.equispaced)
    col = basix.create_element(P, basix.CellType.triangle, 2, basix.LagrangeVariant.equispaced)
    expected = (row.tabulate(0, points)[0, :, :, 0].T * weights) @ col.tabulate(0, points)[
        0, :, :, 0
    ]
    assert np.allclose(evaluate(np.empty(0), np.empty(0), xyz), expected, atol=1e-12)


def boundary_mesh(D):
    from testutil import side_domain

    cell = D.topology.cell_types[0].cell_type
    dim = D.dimension
    facets = basix.cell.topology(cell)[dim - 1]
    facet_type = basix.cell.sub_entity_type(cell, dim - 1, 0)
    S = grouped_topology(
        (basix_cell(facet_type),), ((tuple(facets[0]),),), ((("inside", D.topology, 0),),)
    )
    return side_domain(D, S, "inside")


def boundary_model(cell, degree, coordinate_degree, kind):
    from testutil import normal, side_map

    from uflx_spec_interpreter import pull_back

    _, D, (a, b), (c,) = model(cell, degree, coordinate_degree, "mass")
    dim = D.dimension
    boundary = boundary_mesh(D)
    restriction = side_map(D, boundary, "inside")
    V = a.space
    u, v = function(V, "u"), function(V, "v")
    U, W, A, B = (push_forward(f, D, V.law) for f in (u, v, a, b))
    weight = pull_back(A + num(2) * B + c, restriction)
    match kind:
        case "mass":
            integrand, arguments = weight * pull_back(W * U, restriction), (v, u)
        case "load":
            integrand, arguments = weight * pull_back(W, restriction), (v,)
        case "energy":
            integrand, arguments = weight * weight, ()
        case "normal_derivative":
            g, n = pull_back(gradient(U, D), restriction), normal(restriction)
            flux = sum((g[i] * n[i] for i in range(dim)), num(0))
            integrand, arguments = weight * flux * pull_back(W, restriction), (v, u)
    f = integral(integrand, boundary)
    return (form(f, arguments) if arguments else f), D, (a, b), (c,)


def boundary_reference(cell, degree, coordinate_degree, kind):
    import basix.ufl
    import ufl

    dim = len(basix.cell.topology(cell)) - 1
    coordinate = basix.ufl.element(
        "P", cell, coordinate_degree, basix.LagrangeVariant.equispaced, shape=(dim,)
    )
    D = ufl.Mesh(coordinate)
    V = ufl.FunctionSpace(D, basix.ufl.element("P", cell, degree, basix.LagrangeVariant.equispaced))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a, b, c = ufl.Coefficient(V), ufl.Coefficient(V), ufl.Constant(D)
    weight = a + 2 * b + c
    expressions = {
        "mass": weight * v * u,
        "load": weight * v,
        "energy": weight**2,
        "normal_derivative": weight * ufl.dot(ufl.grad(u), ufl.FacetNormal(D)) * v,
    }
    return expressions[kind] * ufl.ds(metadata={"quadrature_degree": 6})


@pytest.mark.parametrize(
    "cell,degree,coordinate_degree",
    [
        (basix.CellType.triangle, 1, 1),
        (basix.CellType.triangle, 3, 2),
        (basix.CellType.quadrilateral, 2, 2),
        (basix.CellType.tetrahedron, 2, 2),
        (basix.CellType.hexahedron, 1, 1),
    ],
)
@pytest.mark.parametrize("kind", ["energy", "load", "mass", "normal_derivative"])
def test_exterior_facets_match_ffcx(tmp_path, ufcx_include, cell, degree, coordinate_degree, kind):
    jit = pytest.importorskip("ffcx.codegeneration.jit")
    f, D, coefficients, constants = boundary_model(cell, degree, coordinate_degree, kind)
    code = compile_ufcx(f, D, 6, coefficients=coefficients, constants=constants)
    assert code.integral_type == "exterior_facet"
    evaluate, lib = build(code, tmp_path, ufcx_include)
    uf = boundary_reference(cell, degree, coordinate_degree, kind)
    forms, module, _ = jit.compile_forms(
        [uf], options={"scalar_type": "float64"}, cache_dir=tmp_path / "ffcx"
    )
    reference = forms[0].form_integrals[0]
    ffi = module.ffi
    E = basix.create_element(basix.ElementFamily.P, cell, degree, basix.LagrangeVariant.equispaced)
    X = basix.create_element(
        basix.ElementFamily.P, cell, coordinate_degree, basix.LagrangeVariant.equispaced
    )
    rng = np.random.default_rng(31)
    w, c = rng.normal(size=2 * E.dim), np.array([0.7])
    dim = X.points.shape[1]
    xyz = np.zeros((X.dim, 3))
    xyz[:, :dim] = 1.2 * X.points + 0.1
    xyz[:, 0] += 0.15 * X.points[:, 1]
    if coordinate_degree > 1:
        xyz[:, 0] += 0.09 * X.points[:, 0] * X.points[:, 1]
    for facet in range(len(basix.cell.topology(cell)[dim - 1])):
        expected = np.zeros(code.tensor_shape or ())
        reference.tabulate_tensor_float64(
            *(ffi.cast("double*", v.ctypes.data) for v in (expected, w, c, xyz)),
            ffi.new("int[]", [facet]),
            ffi.NULL,
            ffi.NULL,
        )
        assert np.allclose(evaluate(w, c, xyz, facet), expected, rtol=1e-10, atol=1e-11), facet
    address = ctypes.addressof(ctypes.c_char.in_dll(lib, code.name))
    descriptor = ffi.cast("ufcx_form*", address)
    assert [descriptor.form_integral_offsets[i] for i in range(6)] == [0, 0, 1, 1, 1, 1]
    assert descriptor.form_integrals[0].domain == cell.value
    assert not descriptor.form_integrals[0].needs_facet_permutations


@pytest.mark.parametrize("zero", [False, True])
def test_exterior_vertex_geometry_and_zero(tmp_path, ufcx_include, zero):
    from testutil import vertex_mesh

    K = basix_cell(basix.CellType.triangle)
    T = grouped_topology((K,), (((0, 1, 2),),), ())
    D = vertex_mesh(T, [(0, 0), (1, 0), (0, 1)])
    boundary = boundary_mesh(D)
    code = compile_ufcx(integral(num(0 if zero else 1), boundary), D, 2)
    assert code.integral_type == "exterior_facet"
    evaluate, _ = build(code, tmp_path, ufcx_include)
    xyz = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]])
    for facet, vertices in enumerate(basix.cell.topology(basix.CellType.triangle)[1]):
        expected = 0 if zero else np.linalg.norm(xyz[vertices[1]] - xyz[vertices[0]])
        assert np.isclose(evaluate(np.empty(0), np.empty(0), xyz, facet), expected)


def test_cell_and_facet_integrals_require_separate_compilation():
    f, D = simple_form()
    mixed = f.expression + integral(num(1), boundary_mesh(D))
    with pytest.raises(NotImplementedError, match="one integration domain"):
        compile_ufcx(mixed, D, 2)


def interior_mesh(D):
    from testutil import side_domain, tabulated_topology

    from uflx_toy.inclusions import vertex_rows

    assert len(D.topology.cell_types) == 2
    K = D.topology.cell_types[0]
    facet = basix.cell.topology(K.cell_type)[K.dimension - 1][0]
    F = basix_cell(basix.cell.sub_entity_type(K.cell_type, K.dimension - 1, 0))
    points = vertex_rows(K, facet)
    S = tabulated_topology(
        "interior",
        (F,),
        ((("+", D.topology, 0), ("-", D.topology, 1)),),
        ((points, points),),
    )
    return side_domain(D, S, "+")


def interior_model(cell, degree, coordinate_degree, kind):
    from testutil import normal, side_map

    from uflx_spec_interpreter import pull_back

    _, D, (a, b), (c,) = model(cell, degree, coordinate_degree, "mass", cells=2)
    boundary = interior_mesh(D)
    plus, minus = side_map(D, boundary, "+"), side_map(D, boundary, "-")
    V = a.space
    u, v = function(V, "u"), function(V, "v")
    U, W, A, B = (push_forward(f, D, V.law) for f in (u, v, a, b))

    def average(x):
        return num(0.5) * (pull_back(x, plus) + pull_back(x, minus))

    def jump(x):
        return pull_back(x, plus) - pull_back(x, minus)

    weight = average(A) + num(2) * average(B) + c
    match kind:
        case "energy":
            value, arguments = weight * weight, ()
        case "load":
            value, arguments = weight * jump(W), (v,)
        case "mass":
            value, arguments = weight * jump(U) * jump(W), (v, u)
        case "flux":
            gu, gv, n = average(gradient(U, D)), average(gradient(W, D)), normal(plus)
            fluxu = sum((gu[i] * n[i] for i in range(D.dimension)), num(0))
            fluxv = sum((gv[i] * n[i] for i in range(D.dimension)), num(0))
            value = weight * (num(3) * jump(U) * jump(W) - fluxu * jump(W) - fluxv * jump(U))
            arguments = (v, u)
    f = integral(value, boundary)
    return (form(f, arguments) if arguments else f), D, (a, b), (c,)


def interior_reference(cell, degree, coordinate_degree, kind):
    import basix.ufl
    import ufl

    dim = len(basix.cell.topology(cell)) - 1
    coordinate = basix.ufl.element(
        "P", cell, coordinate_degree, basix.LagrangeVariant.equispaced, shape=(dim,)
    )
    D = ufl.Mesh(coordinate)
    V = ufl.FunctionSpace(D, basix.ufl.element("P", cell, degree, basix.LagrangeVariant.equispaced))
    u, v = ufl.TrialFunction(V), ufl.TestFunction(V)
    a, b, c = ufl.Coefficient(V), ufl.Coefficient(V), ufl.Constant(D)
    weight = ufl.avg(a) + 2 * ufl.avg(b) + c
    n = ufl.FacetNormal(D)("+")
    expressions = {
        "energy": weight**2,
        "load": weight * ufl.jump(v),
        "mass": weight * ufl.jump(u) * ufl.jump(v),
        "flux": weight
        * (
            3 * ufl.jump(u) * ufl.jump(v)
            - ufl.dot(ufl.avg(ufl.grad(u)), n) * ufl.jump(v)
            - ufl.dot(ufl.avg(ufl.grad(v)), n) * ufl.jump(u)
        ),
    }
    return expressions[kind] * ufl.dS(metadata={"quadrature_degree": 4})


def paired_geometry(cell, coordinate_element, facets, permutations):
    # Build two conforming physical cells, using FFCx's permutation convention
    # independently of the compiler's own reference tables.
    from ffcx.ir.elementtables import (
        permute_quadrature_interval,
        permute_quadrature_quadrilateral,
        permute_quadrature_triangle,
    )

    dim = coordinate_element.points.shape[1]
    facet_type = basix.cell.sub_entity_type(cell, dim - 1, 0)
    vertices = basix.cell.geometry(cell)
    reference = (
        np.empty((1, 0)) if facet_type == basix.CellType.point else basix.cell.geometry(facet_type)
    )
    permute = {
        basix.CellType.point: lambda points, **kwargs: points,
        basix.CellType.interval: permute_quadrature_interval,
        basix.CellType.triangle: permute_quadrature_triangle,
        basix.CellType.quadrilateral: permute_quadrature_quadrilateral,
    }[facet_type]
    result = []
    for side, (facet, permutation) in enumerate(zip(facets, permutations, strict=True)):
        kwargs = {"reflections": permutation % 2}
        if dim == 3:
            kwargs["rotations"] = permutation // 2
        mapped = permute(reference, **kwargs)
        order = [int(np.argmin(np.linalg.norm(reference - p, axis=1))) for p in mapped]
        face = np.array(basix.cell.topology(cell)[dim - 1][facet])[order]
        source = np.vstack((vertices[face], vertices.mean(axis=0)))
        target = np.zeros_like(source)
        target[:-1, :-1] = reference
        target[-1, :-1] = reference.mean(axis=0)
        target[-1, -1] = 0.8 if side == 0 else -0.6
        transform = np.linalg.lstsq(
            np.column_stack((source, np.ones(len(source)))), target, rcond=None
        )[0]
        xyz = np.zeros((coordinate_element.dim, 3))
        xyz[:, :dim] = (
            np.column_stack((coordinate_element.points, np.ones(coordinate_element.dim)))
            @ transform
        )
        if coordinate_element.degree > 1:
            xyz[:, dim - 1] += 0.06 * xyz[:, 0] * (1 - xyz[:, 0])
        result.append(xyz)
    return np.ascontiguousarray(result)


@pytest.mark.parametrize(
    "cell,degree,coordinate_degree",
    [
        (basix.CellType.interval, 2, 1),
        (basix.CellType.triangle, 1, 1),
        (basix.CellType.triangle, 2, 2),
        (basix.CellType.quadrilateral, 2, 2),
        (basix.CellType.tetrahedron, 1, 1),
        (basix.CellType.hexahedron, 1, 1),
    ],
)
@pytest.mark.parametrize("kind", ["energy", "load", "mass", "flux"])
def test_interior_facets_match_ffcx(tmp_path, ufcx_include, cell, degree, coordinate_degree, kind):
    jit = pytest.importorskip("ffcx.codegeneration.jit")
    f, D, coefficients, constants = interior_model(cell, degree, coordinate_degree, kind)
    code = compile_ufcx(f, D, 4, coefficients=coefficients, constants=constants)
    assert code.integral_type == "interior_facet"
    assert code.coefficients == coefficients
    evaluate, lib = build(code, tmp_path, ufcx_include)
    forms, module, _ = jit.compile_forms(
        [interior_reference(cell, degree, coordinate_degree, kind)],
        options={"scalar_type": "float64"},
        cache_dir=tmp_path / "ffcx",
    )
    ffi = module.ffi
    E = basix.create_element(basix.ElementFamily.P, cell, degree, basix.LagrangeVariant.equispaced)
    X = basix.create_element(
        basix.ElementFamily.P, cell, coordinate_degree, basix.LagrangeVariant.equispaced
    )
    assert code.coefficient_offsets == (0, 2 * E.dim)
    if kind in ("mass", "flux"):
        assert code.tensor_shape == (2 * E.dim, 2 * E.dim)
    rng = np.random.default_rng(94)
    w, c = rng.normal(size=4 * E.dim), np.array([0.4])
    count = len(basix.cell.topology(cell)[D.dimension - 1])
    permutations = {
        basix.CellType.interval: 1,
        basix.CellType.triangle: 2,
        basix.CellType.quadrilateral: 2,
        basix.CellType.tetrahedron: 6,
        basix.CellType.hexahedron: 8,
    }[cell]
    cases = {
        (a, b, (a + b) % permutations, (2 * a + b) % permutations)
        for a in range(count)
        for b in range(count)
    }
    cases |= {
        (p % count, (q + 1) % count, p, q) for p in range(permutations) for q in range(permutations)
    }
    for a, b, p, q in sorted(cases):
        xyz = paired_geometry(cell, X, (a, b), (p, q))
        expected = np.zeros(code.tensor_shape or ())
        forms[0].form_integrals[0].tabulate_tensor_float64(
            *(ffi.cast("double*", v.ctypes.data) for v in (expected, w, c, xyz)),
            ffi.new("int[]", [a, b]),
            ffi.new("uint8_t[]", [p, q]),
            ffi.NULL,
        )
        actual = evaluate(w, c, xyz, (a, b), (p, q))
        assert np.allclose(actual, expected, rtol=1e-9, atol=1e-10), (a, b, p, q)
    address = ctypes.addressof(ctypes.c_char.in_dll(lib, code.name))
    descriptor = ffi.cast("ufcx_form*", address)
    assert [descriptor.form_integral_offsets[i] for i in range(6)] == [0, 0, 0, 1, 1, 1]
    assert descriptor.form_integrals[0].needs_facet_permutations


@pytest.mark.parametrize(
    "cell,degree", [(basix.CellType.triangle, 2), (basix.CellType.tetrahedron, 1)]
)
def test_interior_normals_are_opposite(tmp_path, ufcx_include, cell, degree):
    ufl = pytest.importorskip("ufl")
    jit = pytest.importorskip("ffcx.codegeneration.jit")
    import basix.ufl
    from testutil import normal, side_map

    _, D, _, _ = model(cell, 1, degree, "mass", cells=2)
    boundary = interior_mesh(D)
    a, b = (normal(side_map(D, boundary, side)) for side in ("+", "-"))
    expr = sum((a[i] * b[i] for i in range(D.dimension)), num(0))
    code = compile_ufcx(integral(expr, boundary), D, 4)
    evaluate, _ = build(code, tmp_path, ufcx_include)
    coordinate = basix.ufl.element(
        "P", cell, degree, basix.LagrangeVariant.equispaced, shape=(D.dimension,)
    )
    domain = ufl.Mesh(coordinate)
    reference = -ufl.as_ufl(1) * ufl.dS(domain=domain, metadata={"quadrature_degree": 4})
    forms, module, _ = jit.compile_forms(
        [reference], options={"scalar_type": "float64"}, cache_dir=tmp_path / "ffcx"
    )
    X = basix.create_element(basix.ElementFamily.P, cell, degree, basix.LagrangeVariant.equispaced)
    xyz = paired_geometry(cell, X, (0, 1), (1, 0))
    expected = np.zeros(())
    ffi = module.ffi
    forms[0].form_integrals[0].tabulate_tensor_float64(
        ffi.cast("double*", expected.ctypes.data),
        ffi.NULL,
        ffi.NULL,
        ffi.cast("double*", xyz.ctypes.data),
        ffi.new("int[]", [0, 1]),
        ffi.new("uint8_t[]", [1, 0]),
        ffi.NULL,
    )
    assert np.isclose(evaluate(np.empty(0), np.empty(0), xyz, (0, 1), (1, 0)), expected)


def test_zero_interior_form_keeps_both_argument_blocks(tmp_path, ufcx_include):
    from testutil import side_map

    from uflx_spec_interpreter import pull_back

    f, D = simple_form(zero=True, cells=2)
    boundary = interior_mesh(D)
    trace = pull_back(f.expression.expression, side_map(D, boundary, "+"))
    code = compile_ufcx(form(integral(trace, boundary), f.arguments), D, 2)
    assert code.tensor_shape == (6, 6) and code.integral_type == "interior_facet"
    evaluate, _ = build(code, tmp_path, ufcx_include)
    assert np.array_equal(
        evaluate(np.empty(0), np.empty(0), np.zeros((2, 3, 3)), (0, 1), (0, 1)), np.zeros((6, 6))
    )


def test_interior_piola_normal_jumps(tmp_path, ufcx_include):
    ufl = pytest.importorskip("ufl")
    jit = pytest.importorskip("ffcx.codegeneration.jit")
    import basix.ufl
    from testutil import normal, side_map

    from uflx_spec_interpreter import pull_back

    _, D = simple_form(cells=2)
    interface = interior_mesh(D)
    V = basix_space("RT", D.topology, basix.ElementFamily.RT, 1)
    u, v = function(V, "u"), function(V, "v")

    def jump(f):
        terms = []
        for side in ("+", "-"):
            mapping = side_map(D, interface, side)
            value, n = pull_back(push_forward(f, D, V.law), mapping), normal(mapping)
            terms.append(value[0] * n[0] + value[1] * n[1])
        return terms[0] + terms[1]

    code = compile_ufcx(form(integral(jump(u) * jump(v), interface), (v, u)), D, 4)
    evaluate, _ = build(code, tmp_path, ufcx_include)
    X = basix.create_element(
        basix.ElementFamily.P, basix.CellType.triangle, 1, basix.LagrangeVariant.equispaced
    )
    domain = ufl.Mesh(
        basix.ufl.element("P", "triangle", 1, basix.LagrangeVariant.equispaced, shape=(2,))
    )
    space = ufl.FunctionSpace(domain, basix.ufl.element("RT", "triangle", 1))
    trial, test = ufl.TrialFunction(space), ufl.TestFunction(space)
    n = ufl.FacetNormal(domain)
    reference = ufl.jump(trial, n) * ufl.jump(test, n) * ufl.dS(metadata={"quadrature_degree": 4})
    forms, module, _ = jit.compile_forms(
        [reference], options={"scalar_type": "float64"}, cache_dir=tmp_path / "ffcx"
    )
    ffi = module.ffi
    for facets, permutations in [((0, 2), (0, 1)), ((1, 0), (1, 0))]:
        xyz = paired_geometry(basix.CellType.triangle, X, facets, permutations)
        expected = np.zeros(code.tensor_shape)
        forms[0].form_integrals[0].tabulate_tensor_float64(
            ffi.cast("double*", expected.ctypes.data),
            ffi.NULL,
            ffi.NULL,
            ffi.cast("double*", xyz.ctypes.data),
            ffi.new("int[]", facets),
            ffi.new("uint8_t[]", permutations),
            ffi.NULL,
        )
        assert np.allclose(
            evaluate(np.empty(0), np.empty(0), xyz, facets, permutations), expected, atol=1e-12
        )


@pytest.mark.parametrize("degree", [0, 2])
@pytest.mark.parametrize("geometry_degree", [1, 2])
@pytest.mark.parametrize("action", [False, True])
def test_hhj_compliance(tmp_path, ufcx_include, degree, geometry_degree, action):
    """Double contravariant tensors retain symmetry, ordering and determinant powers."""
    cell = basix.CellType.triangle
    _, D, _, _ = model(cell, 1, geometry_degree, "mass")
    V = basix_space("HHJ", D.topology, basix.ElementFamily.HHJ, degree)
    sigma, tau = function(V, "sigma"), function(V, "tau")
    S, T = (push_forward(f, D, V.law) for f in (sigma, tau))
    density = num(2) * sum((S[i, j] * T[i, j] for i in range(2) for j in range(2)), num(0))
    density += (S[0, 0] + S[1, 1]) * (T[0, 0] + T[1, 1])
    quadrature_degree = 2 * degree + 2 * geometry_degree + 2
    code = compile_ufcx(
        form(integral(density, D), (tau,) if action else (tau, sigma)),
        D,
        quadrature_degree,
        coefficients=(sigma,) if action else (),
    )
    run, _ = build(code, tmp_path, ufcx_include)
    E = basix.create_element(basix.ElementFamily.HHJ, cell, degree)
    X = basix.create_element(
        basix.ElementFamily.P, cell, geometry_degree, basix.LagrangeVariant.equispaced
    )
    points, weights = basix.make_quadrature(cell, quadrature_degree)
    reference = E.tabulate(0, points)[0].reshape(len(points), E.dim, 2, 2)
    xt = X.tabulate(1, points)
    dx = np.stack([xt[d, :, :, 0] for d in (1, 2)], axis=-1)
    coefficients = np.random.default_rng(182).normal(size=E.dim)
    for sign in (-1, 1):
        coordinates = X.points @ np.array([[sign * 1.2, 0.1], [0.05, 0.9]]).T
        if geometry_degree == 2:
            coordinates[:, 0] += 0.08 * X.points[:, 0] * X.points[:, 1]
            coordinates[:, 1] += 0.06 * X.points[:, 0] * (1 - X.points[:, 0])
        J = np.einsum("ai,qaj->qij", coordinates, dx)
        det = np.linalg.det(J)
        physical = np.einsum("qik,qakl,qjl->qaij", J, reference, J) / det[:, None, None, None] ** 2
        mapped = 2 * physical + np.trace(physical, axis1=-2, axis2=-1)[:, :, None, None] * np.eye(2)
        matrix = np.einsum("q,qaij,qbij->ab", weights * abs(det), physical, mapped)
        xyz = np.zeros((X.dim, 3))
        xyz[:, :2] = coordinates
        actual = run(coefficients if action else np.empty(0), np.empty(0), xyz)
        expected = matrix @ coefficients if action else matrix
        np.testing.assert_allclose(actual, expected, rtol=1e-11, atol=1e-11)


def explicit_pair(plus_slot=0):
    """Two independent cells; coincident geometry does not identify their fields."""
    from testutil import side_domain, static_topology, tabulated_topology, vertex_mesh

    from uflx_toy.inclusions import vertex_rows

    K, F = basix_cell(basix.CellType.triangle), basix_cell(basix.CellType.interval)
    T = static_topology((K, K), ((0, 1, 2), (3, 4, 5)), ())
    D = vertex_mesh(T, [(0, 0), (1, 0), (0, 1)] * 2)
    points = vertex_rows(K, (1, 2))
    S = tabulated_topology(
        "explicit-pair",
        (F,),
        ((("+", T, plus_slot), ("-", T, 1 - plus_slot)),),
        ((points, points),),
    )
    return D, side_domain(D, S, "+")


@pytest.mark.parametrize("plus_slot", [0, 1])
def test_one_sided_kernel_preserves_declared_cell_reads(tmp_path, ufcx_include, plus_slot):
    from testutil import side_map

    from uflx_compiler import assemble
    from uflx_spec_interpreter import cell_map, expand_pull_backs, pull_back

    D, E = explicit_pair(plus_slot)
    V = basix_space("DG", D.topology, basix.ElementFamily.P, 1, discontinuous=True)
    u, v = function(V, "u"), function(V, "v")
    U, W = (push_forward(f, D, V.law) for f in (u, v))
    plus, minus = side_map(D, E, "+"), side_map(D, E, "-")
    plus_read = expand_pull_backs(pull_back(pull_back(U, plus), cell_map(E, 0)))
    minus_read = expand_pull_backs(pull_back(pull_back(U, minus), cell_map(E, 0)))
    assert plus_read is not minus_read
    f = form(integral(pull_back(U, plus) * pull_back(W, plus), E), (v, u))
    expected = assemble(f, 2)({})
    code = compile_ufcx(f, D, 2)
    assert code.tensor_shape == expected.shape == (6, 6)
    evaluate, _ = build(code, tmp_path, ufcx_include)
    xyz = np.array([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]] * 2)
    actual = evaluate(np.empty(0), np.empty(0), xyz, (0, 0), (0, 0))
    np.testing.assert_allclose(actual, expected, atol=1e-12)
    unused = slice(3 * (1 - plus_slot), 3 * (2 - plus_slot))
    assert np.all(actual[unused, :] == 0) and np.all(actual[:, unused] == 0)
    assert np.linalg.norm(actual) > 0


def test_two_names_for_one_relation_do_not_manufacture_a_second_cell():
    from testutil import side_domain, tabulated_topology

    from uflx_toy.inclusions import vertex_rows

    _, D = simple_form()
    K = D.topology.cell_types[0]
    F = basix_cell(basix.CellType.interval)
    points = vertex_rows(K, (1, 2))
    S = tabulated_topology(
        "aliased-sides",
        (F,),
        ((("+", D.topology, 0), ("-", D.topology, 0)),),
        ((points, points),),
    )
    E = side_domain(D, S, "+")
    code = compile_ufcx(integral(num(0), E), D, 2)
    assert code.integral_type == "exterior_facet"


def test_a_single_relation_keeps_both_declared_parent_cells():
    from testutil import side_domain, tabulated_topology

    from uflx_toy.inclusions import vertex_rows

    D, _ = explicit_pair()
    K = D.topology.cell_types[0]
    F = basix_cell(basix.CellType.interval)
    S = tabulated_topology(
        "one-side-two-cells",
        (F,),
        ((("inside", D.topology, 0),),),
        ((vertex_rows(K, (1, 2)),),),
    )
    E = side_domain(D, S, "inside")
    V = basix_space("unused", D.topology, basix.ElementFamily.P, 1)
    u, v = function(V, "u"), function(V, "v")
    code = compile_ufcx(form(integral(num(0), E), (v, u)), D, 2)
    assert code.integral_type == "interior_facet"
    assert code.tensor_shape == (6, 6)


def test_facet_kernel_cannot_drop_integration_slots():
    from testutil import side_domain, tabulated_topology

    from uflx_toy.inclusions import vertex_rows

    _, D = simple_form()
    K = D.topology.cell_types[0]
    F = basix_cell(basix.CellType.interval)
    S = tabulated_topology(
        "two-facets",
        (F, F),
        ((("inside", D.topology, 0),),) * 2,
        ((vertex_rows(K, (1, 2)),), (vertex_rows(K, (0, 2)),)),
    )
    with pytest.raises(NotImplementedError, match="one integration cell slot"):
        compile_ufcx(integral(num(0), side_domain(D, S, "inside")), D, 2)


@pytest.mark.parametrize("kind", ["cell", "exterior", "interior"])
@pytest.mark.parametrize("coordinate_degree", [1, 2])
def test_vector_geometry_matches_scalar_components(tmp_path, ufcx_include, kind, coordinate_degree):
    from uflx_spec_interpreter._src.algorithms.substitution import substitute

    make = {"cell": model, "exterior": boundary_model, "interior": interior_model}[kind]
    integrand = {"cell": "stiffness", "exterior": "normal_derivative", "interior": "flux"}[kind]
    f, D, coefficients, constants = make(basix.CellType.triangle, 1, coordinate_degree, integrand)
    X = basix_space("vector-X", D.topology, basix.ElementFamily.P, coordinate_degree, shape=(2,))
    x = function(X, "coordinates")
    mapping = {D: image(x, RD(2)), D.coordinate_function: x}
    mapping.update({c: x[i] for i, c in enumerate(D.coordinate_function.entries)})
    vector_form = substitute(f, mapping)
    options = {"coefficients": coefficients, "constants": constants}
    scalar_code = compile_ufcx(f, D, 5, name="scalar_coordinates", **options)
    vector_code = compile_ufcx(vector_form, mapping[D], 5, name="vector_coordinates", **options)
    assert vector_code.coefficients == coefficients
    scalar_run, _ = build(scalar_code, tmp_path / "scalar", ufcx_include)
    vector_run, _ = build(vector_code, tmp_path / "vector", ufcx_include)
    width = 2 if kind == "interior" else 1
    E = X.element(0)
    xyz = np.zeros((width, E.dim, 3))
    xyz[:, :, :2] = E.points @ np.array([[1.2, 0.1], [0.2, 0.9]])
    xyz[:, :, 0] += 0.06 * E.points[:, 0] * E.points[:, 1]
    if width == 2:
        xyz[1, :, 0] *= -1
    w = np.random.default_rng(32).normal(size=sum(c.space.num_dofs for c in coefficients))
    c = np.array([0.7])
    facets = None if kind == "cell" else ((0, 1) if width == 2 else 2)
    permutations = (0, 1) if width == 2 else None
    np.testing.assert_allclose(
        vector_run(w, c, xyz, facets, permutations),
        scalar_run(w, c, xyz, facets, permutations),
        rtol=1e-11,
        atol=1e-11,
    )


@pytest.mark.parametrize("interior", [False, True])
def test_runtime_facets_preserve_explicit_normal_orientation(tmp_path, ufcx_include, interior):
    from testutil import side_map

    from uflx_spec_interpreter import normal
    from uflx_toy.sides import outward_conormals

    _, D, _, _ = model(basix.CellType.triangle, 1, 1, "mass", cells=2 if interior else 1)
    E = interior_mesh(D) if interior else boundary_mesh(D)
    relation = side_map(D, E, "+" if interior else "inside")
    (m,) = outward_conormals(relation)
    n, reversed_n = normal(relation, m), normal(relation, -m)
    assert n.side_map is reversed_n.side_map
    runs = []
    for label, field in (("outward", n), ("inward", reversed_n)):
        code = compile_ufcx(integral(field[0], E), D, 2, name=label)
        runs.append(build(code, tmp_path / label, ufcx_include)[0])
    xyz = np.array([[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 3.0, 0.0]]] * (2 if interior else 1))
    for facet in range(3):
        entity = (facet, facet) if interior else facet
        permutations = (0, 0) if interior else None
        values = [run(np.empty(0), np.empty(0), xyz, entity, permutations) for run in runs]
        np.testing.assert_allclose(values[0], -values[1], atol=1e-12)
        if facet == 0:
            assert abs(values[0]) > 0


def test_runtime_facet_binding_rejects_a_cut_inclusion():
    from uflx_spec_interpreter import coordinate, domain_map, side_domain
    from uflx_toy.topology import tabulated_topology

    _, D = simple_form()
    K = D.topology.cell_types[0]
    F = basix_cell(basix.CellType.interval)
    S = tabulated_topology("cut", (F,))
    t = coordinate(F)[0]
    A = domain_map(F, tensor((num(0.5) * (num(1) - t), num(0.5) * t)), K)
    E = side_domain(D, S, (0, A))
    with pytest.raises(NotImplementedError, match="complete reference-facet"):
        compile_ufcx(integral(num(1), E), D, 2)
