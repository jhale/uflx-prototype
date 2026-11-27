"""Compile real UFLx cell forms for DOLFINx assembly."""

from hashlib import sha256
from importlib.util import module_from_spec, spec_from_file_location
from tempfile import TemporaryDirectory

import jax
import numpy as np
from cffi import FFI
from dolfinx import fem

import uflx_jax_compiler
from uflx.algorithms import bottom_up
from uflx.forms import form as symbolic_form
from uflx.passes import expand_pull_backs

from .functions import Constant, Function, _ReferenceSpace, functionspace


def _emit_tabulate_tensor(expression, coordinates, dtype, *, coefficients=(), constants=()):
    evaluate = uflx_jax_compiler.trace(expression, coordinates, *coefficients, *constants)
    tabulate = uflx_jax_compiler.with_ufcx_cell_inputs(evaluate, dtype=dtype)
    exported = jax.export.export(jax.jit(tabulate))(*tabulate.input_specs)
    return uflx_jax_compiler.emit_ufcx_tabulate_tensor(exported, "tabulate_tensor")


def _compile_kernel(source, dtype):
    module_name = "_uflx_" + sha256(source.encode()).hexdigest()
    ffi = FFI()
    scalar = {np.dtype(np.float32): "float", np.dtype(np.float64): "double"}[dtype]
    ffi.cdef(
        f"void tabulate_tensor({scalar} *A, const {scalar} *w, const {scalar} *c, "
        f"const {scalar} *coordinate_dofs,"
        " const int *entity_local_index, const uint8_t *quadrature_permutation, void *custom_data);"
    )
    ffi.set_source(module_name, source)
    with TemporaryDirectory() as build_dir:
        library = ffi.compile(tmpdir=build_dir)
        spec = spec_from_file_location(module_name, library)
        module = module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


def _create_form(module, spaces, *, coefficients=(), constants=()):
    domain = spaces[0].mesh
    address = int(module.ffi.cast("uintptr_t", module.ffi.addressof(module.lib, "tabulate_tensor")))
    cells = np.arange(domain.topology.index_map(domain.topology.dim).size_local, dtype=np.int32)
    active_coeffs = np.arange(len(coefficients), dtype=np.int8)
    integrals = {fem.IntegralType.cell: [(0, address, cells, active_coeffs)]}
    cpp_form = fem.form_cpp_class(domain.geometry.x.dtype)(
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


def form(expression, function_spaces=None, *, constant_map=None):
    """Compile a quadrature-discretised cell form with DOLFINx test and trial spaces."""
    symbolic_spaces = expression.argument_spaces
    if not symbolic_spaces:
        raise ValueError("expected at least one form argument")
    if function_spaces is None:
        function_spaces = []
        for space in symbolic_spaces:
            reference = getattr(space, "reference_space", None)
            if not isinstance(reference, _ReferenceSpace):
                raise ValueError("expected a form argument declared on a DOLFINx function space")
            function_spaces.append(reference.function_space)
    elif tuple(functionspace(V) for V in function_spaces) != symbolic_spaces:
        raise ValueError("expected a matching DOLFINx space for each form argument")
    if expression.value_shape or expression.free_indices:
        raise ValueError("assembly requires a scalar form with no free indices")

    mesh = function_spaces[0].mesh
    dtype = mesh.geometry.x.dtype
    if dtype not in (np.float32, np.float64):
        raise TypeError("cell assembly requires float32 or float64 geometry")
    for V in function_spaces:
        if V.mesh is not mesh or V.element.dtype != dtype:
            raise ValueError("argument spaces must share the assembly mesh and its precision")
    if jax.config.x64_enabled != (dtype == np.float64):
        raise ValueError(f"set JAX_ENABLE_X64={int(dtype == np.float64)} for {dtype} assembly")
    constants = {} if constant_map is None else dict(constant_map)

    for symbol, value in constants.items():
        if value.dtype != dtype:
            raise ValueError(f"constant {symbol.label} has dtype {value.dtype}, expected {dtype}")
        if value.value.shape != symbol.value_shape:
            raise ValueError("constant shape does not match the cell kernel")
    coefficients = {}

    def bind(node):
        if isinstance(node, Constant):
            if node.data.dtype != dtype:
                raise ValueError(f"constant has dtype {node.data.dtype}, expected {dtype}")
            constants[node.symbol] = node.data
            return node.symbol
        if isinstance(node, Function):
            if node.data.function_space.mesh is not mesh:
                raise ValueError("coefficient and form must use the same DOLFINx mesh")
            if node.data.dtype != dtype:
                raise ValueError(
                    f"coefficient {node.data.name} has dtype {node.data.dtype}, expected {dtype}"
                )
            coefficients[node.symbol] = node.data
            return node.symbol
        return node

    integrand = bottom_up(expression.expression, bind)
    # Binding a live field exposes its mapped symbol in existing cell pull-backs.
    discrete = symbolic_form(
        expand_pull_backs(integrand) if coefficients else integrand, expression.arguments
    )
    coordinates = symbolic_spaces[0].domain.coordinate_function
    source = _emit_tabulate_tensor(
        discrete, coordinates, dtype, coefficients=coefficients, constants=constants
    )
    module = _compile_kernel(source, dtype)
    return _create_form(
        module, function_spaces, coefficients=coefficients.values(), constants=constants.values()
    )
