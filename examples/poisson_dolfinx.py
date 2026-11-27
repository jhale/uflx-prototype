# %% [markdown]
"""# Poisson with DOLFINx and EmitC

Use P1 solution and coefficient spaces on a unit square with P2 triangle geometry.
"""

# %%
from importlib.util import module_from_spec, spec_from_file_location
from math import prod
from tempfile import TemporaryDirectory

import basix
import jax
import numpy as np
from cffi import FFI
from dolfinx import fem, la, mesh
from dolfinx.la.superlu_dist import superlu_dist_matrix, superlu_dist_solver
from mpi4py import MPI

import uflx_jax_compiler
from uflx.calculus import gradient, integral
from uflx.forms import form, function
from uflx.geometry import create_mesh, euclidean_space
from uflx.passes import lower
from uflx_basix.cells import topology
from uflx_basix.quadrature import apply_quadrature
from uflx_basix.spaces import function_space, local_space, real_scalar_space

jax.config.update("jax_enable_x64", True)

# There is no real coupling between UFLx and DOLFINx, this would between done
# later.
triangle = basix.CellType.triangle
T = topology(triangle)
p2 = basix.create_element(basix.ElementFamily.P, triangle, 2)
x = function(local_space(T, p2, shape=(2,)), "x")
domain = create_mesh(x, euclidean_space(2))

p1 = basix.create_element(basix.ElementFamily.P, triangle, 1)
V = function_space(domain, p1)
u, v = function(V, "u"), function(V, "v")

kappa = function(V, "kappa")
stiffness_integral = integral(kappa * (gradient(v, domain) @ gradient(u, domain)), domain)

f = function(real_scalar_space(), "f")
load_integral = integral(f * v, domain)

rule = basix.make_quadrature(triangle, 1)
a = form(apply_quadrature(lower(stiffness_integral), *rule), (v, u))
L = form(apply_quadrature(lower(load_integral), *rule), (v,))


# %%
def emit_cell_kernel(expression, coordinates, *, coefficients=(), constants=()):
    evaluate = uflx_jax_compiler.trace(expression, coordinates, *coefficients, *constants)
    tabulate = jax.jit(uflx_jax_compiler.with_ufcx_cell_inputs(evaluate, dtype=np.float64))
    sizes = (
        3 * coordinates.space.element(0).dim,
        sum(field.space.reference_space.num_dofs for field in coefficients),
        sum(prod(value.value_shape) for value in constants),
    )
    inputs = [jax.ShapeDtypeStruct((n,), np.float64) for n in sizes]
    exported = jax.export.export(tabulate)(*inputs)
    return uflx_jax_compiler.emit_ufcx_tabulate_tensor(exported, "tabulate_tensor")


def compile_kernel(source, module_name):
    ffi = FFI()
    ffi.cdef(
        "void tabulate_tensor(double *, const double *, const double *, const double *,"
        " const int *, const uint8_t *, void *);"
    )
    ffi.set_source(module_name, source)
    with TemporaryDirectory() as build_dir:
        library = ffi.compile(tmpdir=build_dir)
        spec = spec_from_file_location(module_name, library)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


# %%
def dolfinx_cell_form(module, spaces, *, coefficients=(), constants=()):
    domain = spaces[0].mesh
    address = int(module.ffi.cast("uintptr_t", module.ffi.addressof(module.lib, "tabulate_tensor")))
    cells = np.arange(domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32)
    active_coeffs = np.arange(len(coefficients), dtype=np.int8)
    integrals = {fem.IntegralType.cell: [(0, address, cells, active_coeffs)]}
    cpp_form = fem.form_cpp_class(np.float64)(
        [space._cpp_object for space in spaces],
        integrals,
        [field._cpp_object for field in coefficients],
        [value._cpp_object for value in constants],
        False,
        [],
        mesh=domain._cpp_object,
    )
    # The form holds the extension module for the lifetime of the function pointer.
    return fem.Form(cpp_form, domain, spaces, module=module)


square = mesh.create_unit_square(MPI.COMM_WORLD, 8, 8)
square = fem.interpolate_geometry(square, fem.coordinate_element(p2))
P1 = fem.functionspace(square, ("Lagrange", 1))

kappa_h = fem.Function(P1)
kappa_h.interpolate(lambda X: 3 - 2 * X[0])
kappa_h.x.scatter_forward()
stiffness_kernel = compile_kernel(
    emit_cell_kernel(a, x, coefficients=(kappa,)), "_uflx_stiffness"
)
stiffness = dolfinx_cell_form(stiffness_kernel, [P1, P1], coefficients=(kappa_h,))

f_h = fem.Constant(square, np.float64(2))
load_kernel = compile_kernel(emit_cell_kernel(L, x, constants=(f,)), "_uflx_load")
load = dolfinx_cell_form(load_kernel, [P1], constants=(f_h,))

boundary = fem.locate_dofs_geometrical(P1, lambda X: np.isclose(X[0], 0) | np.isclose(X[0], 1))
boundary_values = fem.Function(P1)
boundary_values.interpolate(lambda X: X[0])
bc = fem.dirichletbc(boundary_values, boundary)

A = fem.assemble_matrix(stiffness, bcs=[bc])
A.scatter_reverse()
b = fem.assemble_vector(load)
fem.apply_lifting(b.array, [stiffness], bcs=[[bc]])
b.scatter_reverse(la.InsertMode.add)
bc.set(b.array)

solution = fem.Function(P1)
solver = superlu_dist_solver(superlu_dist_matrix(A))
info = solver.solve(b, solution.x)
if info != 0:
    raise RuntimeError("SuperLU_DIST solve failed.")
solution.x.scatter_forward()

x_nodes = P1.tabulate_dof_coordinates()[:, 0]
np.testing.assert_allclose(solution.x.array, x_nodes, atol=1e-12, rtol=0)
