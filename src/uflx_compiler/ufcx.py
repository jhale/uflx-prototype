"""Float64 cell and facet forms with UFCx-local inputs."""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import basix
import numpy as np

from uflx.algorithms import (
    bottom_up,
    derived_index,
    derived_indices,
    map_fields,
    memoized,
    operands,
    replace,
    substitute_values,
)
from uflx.arithmetic import add, multiply
from uflx.calculus import reference_integral
from uflx.nodes import (
    AbstractExpression,
    Form,
    Function,
    GatheredIndex,
    Integral,
    Mesh,
    PullBack,
    SideMap,
    Tensor,
    Zero,
)
from uflx.passes import apply_derivatives, expand_applications, lower
from uflx.tensors import index_sum
from uflx_basix.basix_nodes import BasixSpace
from uflx_basix.nodes import IndexTable
from uflx_basix.runtime import runtime_array

from .algebra import _normalizer
from .c_code import c_kernel
from .evaluation import evaluate
from .facets import _facet_tables
from .quadrature import apply_quadrature, constant
from .simplification import simplify
from .sum_factorisation import action_kernel
from .tabulation import arrayed, leaf, nested, separable, table_read


@dataclass(frozen=True)
class UFCxCode:
    """C source and the input ordering needed by an assembler.

    Coefficients use standard Basix DOF ordering, already transformed by the
    assembler. Coordinate DOFs have three interleaved components per scalar DOF.
    Offsets count scalar entries, not bytes. The generated symbols are ``name``
    (ufcx_form), ``name_integral`` and ``name_kernel``.
    """

    source: str
    name: str
    tensor_shape: tuple[int, ...]
    coefficients: tuple[Function, ...]
    constants: tuple[Function, ...]
    coefficient_offsets: tuple[int, ...]
    constant_offsets: tuple[int, ...]
    integral_type: str


def _nodes(expression):
    seen = set()

    def visit(node):
        if node not in seen:
            seen.add(node)
            yield node
            for child in operands(node):
                yield from visit(child)

    return visit(expression)


def _element(space):
    if not isinstance(space, BasixSpace) or len(set(space.topology.cell_types)) != 1:
        raise NotImplementedError("UFCx cell kernels require single-cell-type Basix spaces")
    return basix.create_element(
        space.family,
        space.topology.cell_types[0].cell_type,
        space.degree,
        space.element(0).lagrange_variant,
        discontinuous=space.discontinuous,
    )


def _offsets(sizes):
    result, offset = [], 0
    for size in sizes:
        result.append(offset)
        offset += size
    return tuple(result)


def compile_ufcx(
    f: AbstractExpression,
    mesh: Mesh,
    degree: int,
    *,
    name: str = "uflx_form",
    coefficients: Sequence[Function] | None = None,
    constants: Sequence[Function] | None = None,
    sum_factorisation: bool = False,
) -> UFCxCode:
    """Emit one UFCx cell, exterior-facet or interior-facet integral.

    Supports real scalar functionals and forms of rank one or two on one mesh
    with one Basix cell type. Geometry is a vector Basix function or a tensor of
    scalar Basix functions of the coordinate element. Explicit sequences set input
    order; otherwise construction order is used. Cell and exterior-facet kernels
    require one parent cell slot; interior-facet kernels require two distinct
    parent slots. Explicit side-map entries select these slots. Facet kernels
    require one integration slot. Unsupported configurations are rejected rather
    than truncated or expanded; zero and one-sided integrands keep that layout.
    Blocked scalar elements support vector/tensor arguments and coefficients
    with interleaved components. Mixed arguments are rejected.
    ``sum_factorisation=True`` enables experimental staged tensor-product
    contractions for scalar cell actions. Unsupported forms or schedules exceeding
    256 KiB of temporary storage use the ordinary emitter.
    The caller compiles the returned source with FFCx's ``ufcx.h``.
    """
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None:
        raise ValueError("name must be a C identifier beginning with a letter")
    if f.scalar_type != "real" or f.value_shape or f.free_indices:
        raise NotImplementedError("UFCx cell forms must be real scalars without free indices")
    if len(set(mesh.topology.cell_types)) != 1:
        raise NotImplementedError("UFCx cell kernels require one cell type")
    geometry, coordinate_element = _coordinate_fields(mesh)

    normalize = _normalizer()
    f = apply_derivatives(expand_applications(f))
    domain = _integration_domain(f, mesh)
    # The expression owns the complete local configuration, including unused slots.
    width = len(mesh.topology.cell_types)
    f = normalize(f)
    arguments = ()
    while isinstance(f, Form):
        arguments = f.arguments + arguments
        f = f.expression
    if len(arguments) > 2 or any(not isinstance(a, Function) for a in arguments):
        raise NotImplementedError("UFCx supports at most two non-mixed real arguments")
    functions = tuple(n for n in _nodes(f) if isinstance(n, Function))
    fields = tuple(
        n for n in functions if n not in arguments and n not in geometry and n.domain is not None
    )
    values = tuple(n for n in functions if n.domain is None and n not in arguments)
    coefficients = fields if coefficients is None else tuple(coefficients)
    constants = values if constants is None else tuple(constants)
    for supplied, needed, kind in (
        (coefficients, fields, "coefficients"),
        (constants, values, "constants"),
    ):
        if len(set(supplied)) != len(supplied) or set(supplied) != set(needed):
            raise ValueError(f"{kind} must list each input exactly once")
    if any(u.scalar_type != "real" for u in (*arguments, *coefficients, *constants, *geometry)):
        raise NotImplementedError("UFCx currently supports real-valued inputs only")
    elements = tuple(_element(u.space) for u in (*arguments, *coefficients))
    if any(u.domain != mesh.topology for u in (*arguments, *coefficients)):
        raise NotImplementedError("all fields must belong to the integration mesh topology")
    dimensions = tuple(
        E.dim * u.space.block_size
        for u, E in zip((*arguments, *coefficients), elements, strict=True)
    )
    shape = tuple(width * n for n in dimensions[: len(arguments)])
    coefficient_offsets = _offsets(width * n for n in dimensions[len(arguments) :])
    constant_offsets = _offsets(prod(c.value_shape) for c in constants)
    rows = tuple(derived_index("compiler.ufcx.argument", i) for i in range(len(arguments)))
    components = {
        u: derived_index("compiler.ufcx.component", i)
        for i, u in enumerate(arguments)
        if u.space.block_size > 1
    }
    argument_rows = dict(zip(arguments, rows, strict=True))
    output_indices = tuple(
        index
        for u, row in argument_rows.items()
        for index in ((row, components[u]) if u in components else (row,))
    )
    output_extents = tuple(
        extent
        for u, E in zip(arguments, elements, strict=False)
        for extent in ((width * E.dim, u.space.block_size) if u in components else (width * E.dim,))
    )
    inputs = {
        u: runtime_array(("w", off), (width, E.dim, u.space.block_size))
        for u, E, off in zip(
            coefficients, elements[len(arguments) :], coefficient_offsets, strict=True
        )
    }
    inputs.update(
        {
            c: runtime_array(("c", off), c.value_shape)
            for c, off in zip(constants, constant_offsets, strict=True)
        }
    )
    bindings, reference_values = {}, {}
    if domain == mesh:
        integral_type = "cell"
    else:
        integral_type = "interior_facet" if width == 2 else "exterior_facet"
        domain, mapping, reference_values, bindings = _facet_tables(
            tuple(_nodes(f)), mesh, domain, width
        )
        f = replace(f, mapping)
    factor_axes = {}
    if (
        sum_factorisation
        and integral_type == "cell"
        and len(arguments) == 1
        and not arguments[0].value_shape
        and arguments[0].space.factors(0)
    ):
        factor_axes = {
            arguments[0]: derived_indices(
                "compiler.ufcx.factor", (arguments[0],), len(arguments[0].space.factors(0))
            )
        }
    expression = _local_expression(
        f,
        degree,
        argument_rows,
        components,
        geometry,
        inputs,
        coordinate_element.dim,
        domain,
        width,
        reference_values,
        factor_axes,
    )
    kernel = None
    if factor_axes and set(factor_axes[arguments[0]]) == dict(expression.free_indices).keys():
        permutation = np.argsort(arguments[0].space.element(0).dof_ordering).tolist()
        kernel = action_kernel(
            expression,
            name + "_kernel",
            lambda label, off: f"{label[0]}[{label[1]} + ({off})]",
            factor_axes[arguments[0]],
            permutation,
        )
    if kernel is None:
        if factor_axes:
            expression = _local_expression(
                f,
                degree,
                argument_rows,
                components,
                geometry,
                inputs,
                coordinate_element.dim,
                domain,
                width,
                reference_values,
            )
        expected_indices = tuple(zip(output_indices, output_extents, strict=True))
        if set(dict(expression.free_indices)) - (set(output_indices) | bindings.keys()):
            raise ValueError("local tensor has unexpected free indices")
        if set(output_indices) - dict(expression.free_indices).keys():
            expression = add(expression, Zero((), expected_indices, (), "real"))
        kernel = c_kernel(
            expression,
            name + "_kernel",
            lambda label, off: f"{label[0]}[{label[1]} + ({off})]",
            output_indices=output_indices,
            unroll_indices=tuple(components.values()),
            index_bindings=bindings,
        )
    source = _descriptors(
        name, kernel, shape, coefficients, constants, elements, coordinate_element, integral_type
    )
    return UFCxCode(
        source,
        name,
        shape,
        tuple(coefficients),
        tuple(constants),
        coefficient_offsets,
        constant_offsets,
        integral_type,
    )


def _integration_domain(f, mesh):
    domains = {n.integration_domain for n in _nodes(f) if isinstance(n, Integral)}
    if not domains:
        domains = {mesh}  # A domain-free zero form may represent a cell kernel.
    if len(domains) != 1:
        raise NotImplementedError("one integration domain is required per UFCx form")
    (domain,) = domains
    if domain == mesh:
        if len(mesh.topology.cell_types) != 1:
            raise NotImplementedError("UFCx cell kernels require exactly one parent cell slot")
        return domain
    if not isinstance(domain, Mesh) or domain.dimension != mesh.dimension - 1:
        raise NotImplementedError("expected a cell or facet integration domain")
    if len(domain.topology.cell_types) != 1:
        raise NotImplementedError("UFCx facet kernels require exactly one integration cell slot")
    if len(mesh.topology.cell_types) not in (1, 2):
        raise NotImplementedError("UFCx facet kernels require one or two parent cell slots")
    match domain.coordinate_function:
        case PullBack(x, SideMap(T, S, _)) if T == mesh.topology and S == domain.topology:
            if x == mesh.coordinate_function:
                return domain
    raise NotImplementedError("facet geometry must be a side_domain of the supplied mesh")


def _coordinate_fields(mesh):
    x = mesh.coordinate_function
    if isinstance(x, Function) and isinstance(x.space, BasixSpace):
        if x.space.block_shape != (mesh.ambient_domain.dimension,):
            raise NotImplementedError("UFCx vector geometry requires a blocked scalar element")
        geometry = (x,)
    elif isinstance(x, Tensor):
        geometry = x.entries
    else:
        raise NotImplementedError(
            "UFCx geometry requires a vector Basix function or a tensor of scalar functions"
        )
    if not 1 <= mesh.ambient_domain.dimension <= 3 or any(
        not isinstance(x, Function) or x.domain != mesh.topology for x in geometry
    ):
        raise NotImplementedError("UFCx geometry requires one to three coordinate components")
    if isinstance(mesh.coordinate_function, Tensor) and any(x.value_shape for x in geometry):
        raise NotImplementedError("UFCx coordinate tensor entries must be scalar functions")
    coordinate_element = _element(geometry[0].space)
    if any(_element(x.space) != coordinate_element for x in geometry):
        raise ValueError("coordinate components must use the same element")

    return geometry, coordinate_element


def _local_expression(
    f,
    degree,
    argument_rows,
    components,
    geometry,
    inputs,
    coordinate_dimension,
    domain,
    width,
    reference_values,
    factor_axes=None,
):
    coordinate_data = runtime_array(("coordinate_dofs", 0), (width, coordinate_dimension, 3))

    def local_integral(node):
        if not isinstance(node, Integral):
            return node
        if node.integration_domain != domain:
            raise NotImplementedError("one integration domain is required per UFCx form")
        return reference_integral(node.expression, domain, 0)

    f = apply_quadrature(lower(bottom_up(f, local_integral)), degree)
    f = substitute_values(f, reference_values)

    def tabulate(node):
        if node in inputs:
            return inputs[node]
        found = leaf(node)
        if found is None:
            return node
        u, cell, order, points = found
        points = replace(points, reference_values)
        if not isinstance(u, Function) or cell not in range(width):
            raise NotImplementedError("expected a local Basix field read")
        E = _element(u.space)
        dof = derived_index("compiler.ufcx.dof", u, cell)
        block_size = u.space.block_size
        if width == 1 and u in argument_rows:
            dof = argument_rows[u]

        if factor_axes and u.space.factors(0) and separable(points) and not u.value_shape:
            factors = u.space.factors(0)
            axes = factor_axes.get(u) or derived_indices(
                "compiler.ufcx.factor_dof", (u, cell), len(factors)
            )
            permutation = np.argsort(u.space.element(0).dof_ordering).reshape(
                tuple(F.dim for F in factors)
            )
            dofs = IndexTable(nested(permutation), E.dim)
            position = GatheredIndex(dofs, axes)

            def factored_component(directions):
                value = None
                for k, (F, X, axis) in enumerate(zip(factors, points.entries, axes, strict=True)):
                    term = table_read(u.space, 0, F, X, (0,) * directions.count(k), axis, 0)
                    value = term if value is None else multiply(value, term)
                if u in argument_rows:
                    return value
                data = (
                    coordinate_data[cell, position, geometry.index(u)]
                    if u in geometry
                    else inputs[u][cell, position, 0]
                )
                return index_sum(data * value, axes)

            return arrayed((points.value_shape[0],) * order, factored_component)

        def component(idx):
            rank = len(u.value_shape)
            c = 0
            for i, n in zip(idx[:rank], u.value_shape, strict=True):
                c = c * n + i
            component_number = c if u.space.block_shape else 0
            value = table_read(
                u.space, 0, E, points, idx[rank:], dof, 0 if u.space.block_shape else c
            )
            if u in argument_rows:
                if width == 2:
                    value = _argument_block(value, dof, argument_rows[u], cell, E.dim)
                if u in components:
                    selector = constant(tuple(float(k == c) for k in range(block_size)))
                    value = multiply(selector[components[u]], value)
                return value
            data = (
                coordinate_data[cell, dof, c if u.space.block_shape else geometry.index(u)]
                if u in geometry
                else inputs[u][cell, dof, component_number]
            )
            return index_sum(data * value, (dof,))

        return arrayed((*u.value_shape, *(points.value_shape[0],) * order), component)

    # Replace whole field reads before visiting their Function operands.
    def visit(visit, node):
        replacement = tabulate(node)
        return (
            replacement if replacement is not node else map_fields(node, visit, node.value_fields)
        )

    return simplify(memoized(visit)(f))


def _argument_block(value, dof, row, side, dimension):
    # Basis tables stay separate for each side; only their local DOF axis doubles.
    axes = [i for i, _ in value.free_indices]
    values = np.moveaxis(evaluate(value, {}), axes.index(dof), -1)
    blocked = np.zeros((*values.shape[:-1], 2 * dimension))
    blocked[..., side * dimension : (side + 1) * dimension] = values
    indices = tuple(i for i in axes if i != dof)
    return constant(blocked)[(*indices, row)]


def _descriptors(
    name, kernel, shape, coefficients, constants, elements, coordinate_element, integral_type
):
    declarations = []

    def array(suffix, ctype, entries):
        entries = list(entries)
        if not entries:
            return "NULL"
        symbol = name + "_" + suffix
        declarations.append(f"static {ctype} {symbol}[] = {{{', '.join(map(str, entries))}}};")
        return symbol

    enabled = array("enabled", "const bool", ["true"] * len(coefficients))
    positions = array("positions", "int", range(len(coefficients)))
    coefficient_names = array(
        "coefficient_names", "const char*", (json.dumps(str(c.label)) for c in coefficients)
    )
    constant_names = array(
        "constant_names", "const char*", (json.dumps(str(c.label)) for c in constants)
    )
    ranks = array("constant_ranks", "const int", (len(c.value_shape) for c in constants))
    shapes = array(
        "constant_shapes",
        "const int*",
        (array(f"shape_{i}", "const int", c.value_shape) for i, c in enumerate(constants)),
    )
    hashes = array("element_hashes", "uint64_t", (f"UINT64_C({E.hash()})" for E in elements))
    needs_permutations = "true" if integral_type == "interior_facet" else "false"
    declarations.append(
        f"ufcx_integral {name}_integral = {{\n"
        f"  .enabled_coefficients = {enabled}, .tabulate_tensor_float64 = {name}_kernel,\n"
        f"  .needs_facet_permutations = {needs_permutations},\n"
        f"  .coordinate_element_hash = UINT64_C({coordinate_element.hash()}),\n"
        f"  .domain = {coordinate_element.cell_type.value}\n}};"
    )
    integrals = array("integrals", "ufcx_integral*", [f"&{name}_integral"])
    ids = array("ids", "int", [-1])
    kind = ("cell", "exterior_facet", "interior_facet").index(integral_type)
    offsets = array("integral_offsets", "int", [int(i > kind) for i in range(6)])
    signature = hashlib.sha256(kernel.encode()).hexdigest()
    declarations.append(f'''ufcx_form {name} = {{
  .signature = "{signature}", .rank = {len(shape)},
  .num_coefficients = {len(coefficients)}, .original_coefficient_positions = {positions},
  .coefficient_name_map = {coefficient_names}, .num_constants = {len(constants)},
  .constant_ranks = {ranks}, .constant_shapes = {shapes}, .constant_name_map = {constant_names},
  .finite_element_hashes = {hashes}, .form_integrals = {integrals},
  .form_integral_ids = {ids}, .form_integral_offsets = {offsets}
}};''')
    return (
        '#include <stddef.h>\n#include <math.h>\n#include "ufcx.h"\n'
        + kernel
        + "\n"
        + "\n".join(declarations)
        + "\n"
    )
