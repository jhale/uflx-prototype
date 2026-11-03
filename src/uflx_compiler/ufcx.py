"""Float64 cell and facet forms with UFCx-local inputs."""

import hashlib
import json
import re
from collections.abc import Sequence
from dataclasses import dataclass
from math import prod

import basix
import numpy as np

from uflx_spec_interpreter import (
    add,
    apply_actions,
    apply_derivatives,
    function,
    image,
    index_sum,
    lower,
    multiply,
    tensor,
)
from uflx_spec_interpreter._src.names import _derived_index
from uflx_spec_interpreter._src.nodes import (
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
from uflx_spec_interpreter._src.passes.integrals import _piece_integral
from uflx_spec_interpreter._src.substitution import _substitute
from uflx_spec_interpreter._src.traversal import (
    _bottom_up,
    _lowered,
    _map_fields,
    _memoized,
    _operands,
)
from uflx_toy.basix_nodes import BasixSpace
from uflx_toy.basix_spaces import basix_space
from uflx_toy.nodes import VertexCoordinates
from uflx_toy.runtime import runtime_array
from uflx_toy.topology import tabulated_topology

from .algebra import _normalizer
from .c_code import c_kernel
from .evaluation import evaluate
from .facets import _facet_tables
from .quadrature import apply_quadrature, constant
from .simplification import simplify
from .tabulation import arrayed, leaf, table_read


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
            for child in _operands(node):
                yield from visit(child)

    return visit(expression)


def _element(space):
    if not isinstance(space, BasixSpace) or len(space.topology.cell_types) != 1:
        raise NotImplementedError("UFCx cell kernels require single-cell-type Basix spaces")
    return basix.create_element(
        space.family,
        space.topology.cell_types[0].cell_type,
        space.degree,
        basix.LagrangeVariant.equispaced,
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
) -> UFCxCode:
    """Emit one UFCx cell, exterior-facet or interior-facet integral.

    Supports real scalar functionals and forms of rank one or two on one mesh
    with one Basix cell type. Geometry is a tensor of scalar Basix functions or
    vertex coordinates (P1). Explicit coefficient/constant sequences set input
    order; otherwise construction order is used. A side_domain with one side
    selects exterior facets; two sides named + and - select interior facets.
    Blocked scalar elements support vector/tensor arguments and coefficients
    with interleaved components. Mixed arguments are rejected.
    The caller compiles the returned source with FFCx's ``ufcx.h``.
    """
    if re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", name) is None:
        raise ValueError("name must be a C identifier beginning with a letter")
    if f.scalar_type != "real" or f.value_shape or f.free_indices:
        raise NotImplementedError("UFCx cell forms must be real scalars without free indices")
    if len(mesh.topology.cell_types) != 1:
        raise NotImplementedError("UFCx cell kernels require one cell type")
    f, mesh, geometry, coordinate_element = _coordinate_fields(f, mesh)

    originals = {}
    normalize = _normalizer()
    f = apply_derivatives(apply_actions(f))
    domain = _integration_domain(f, mesh)
    width = 1 if domain == mesh else len(domain.topology.sides[0])
    if width == 2:
        # A generic two-cell topology keeps local side 1 valid even for periodic
        # or symbolic input topologies. User inputs are returned in their original identity.
        local = tabulated_topology("ufcx_pair", mesh.topology.cell_types, (2,), (), ())
        mapping = {mesh.topology: local}
        originals = {_substitute(u, mapping): u for u in _nodes(f) if isinstance(u, Function)}
        f, mesh, domain = (_substitute(n, mapping) for n in (f, mesh, domain))
        geometry = tuple(_substitute(n, mapping) for n in geometry)
        if coefficients is not None:
            coefficients = tuple(_substitute(n, mapping) for n in coefficients)
        if constants is not None:
            constants = tuple(_substitute(n, mapping) for n in constants)
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
    rows = tuple(_derived_index("compiler.ufcx.argument", i) for i in range(len(arguments)))
    components = {
        u: _derived_index("compiler.ufcx.component", i)
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
        local, reference_values, bindings = _facet_tables(mesh, domain, width)
        mapping = {domain.topology: local}
        f, domain = (_substitute(n, mapping) for n in (f, domain))
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
        tuple(originals.get(u, u) for u in coefficients),
        tuple(originals.get(u, u) for u in constants),
        coefficient_offsets,
        constant_offsets,
        integral_type,
    )


def _integration_domain(f, mesh):
    domains = {n.integration_domain for n in _nodes(f) if isinstance(n, Integral)}
    if not domains:
        return mesh  # A zero form may have no surviving integral.
    if len(domains) != 1:
        raise NotImplementedError("one integration domain is required per UFCx form")
    (domain,) = domains
    if domain == mesh:
        return domain
    if not isinstance(domain, Mesh) or domain.dimension != mesh.dimension - 1:
        raise NotImplementedError("expected a cell or facet integration domain")
    widths = {len(sides) for sides in domain.topology.sides}
    if widths not in ({1}, {2}) or any(
        parent != mesh.topology or group != 0
        for sides in domain.topology.sides
        for _, parent, group in sides
    ):
        raise NotImplementedError("expected one or two sides on the supplied topology")
    if widths == {2} and any(
        {s[0] for s in sides} != {"+", "-"} for sides in domain.topology.sides
    ):
        raise NotImplementedError("interior facets require sides named '+' and '-'")
    match domain.coordinate_function:
        case PullBack(x, SideMap(T, S, _)) if T == mesh.topology and S == domain.topology:
            if x == mesh.coordinate_function:
                return domain
    raise NotImplementedError("facet geometry must be a side_domain of the supplied mesh")


def _cell_number(cell):
    if isinstance(cell, GatheredIndex) and all(isinstance(i, int) for i in cell.indices):
        values = cell.table.values
        for i in cell.indices:
            values = values[i]
        return values
    return cell


def _coordinate_fields(f, mesh):
    if isinstance(mesh.coordinate_function, VertexCoordinates):
        X = basix_space("ufcx_geometry", mesh.topology, basix.ElementFamily.P, 1)
        geometry = tuple(
            function(X, ("ufcx_geometry", k))
            for k in range(mesh.ambient_coordinate_domain.dimension)
        )
        local_mesh = image(tensor(geometry), mesh.ambient_coordinate_domain)
        f = _substitute(
            f, {mesh: local_mesh, mesh.coordinate_function: local_mesh.coordinate_function}
        )
        mesh = local_mesh
    else:
        if not isinstance(mesh.coordinate_function, Tensor):
            raise NotImplementedError("UFCx geometry must be a tensor of scalar Basix functions")
        geometry = mesh.coordinate_function.entries
    if not 1 <= len(geometry) <= 3 or any(
        not isinstance(x, Function) or x.value_shape or x.domain != mesh.topology for x in geometry
    ):
        raise NotImplementedError("UFCx geometry requires one to three scalar coordinate functions")
    coordinate_element = _element(geometry[0].space)
    if any(_element(x.space) != coordinate_element for x in geometry):
        raise ValueError("coordinate components must use the same element")

    return f, mesh, geometry, coordinate_element


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
):
    coordinate_data = runtime_array(("coordinate_dofs", 0), (width, coordinate_dimension, 3))

    def local_integral(node):
        if not isinstance(node, Integral):
            return node
        if node.integration_domain != domain:
            raise NotImplementedError("one integration domain is required per UFCx form")
        piece = _piece_integral(node.expression, domain, 0)
        return _substitute(piece.expression, {piece.indices[0]: 0})

    f = apply_quadrature(lower(_bottom_up(f, local_integral)), degree)
    f = _substitute(f, reference_values)

    def tabulate(node):
        if node in inputs:
            return inputs[node]
        found = leaf(node)
        if found is None:
            return node
        u, (piece, cell), order, points = found
        cell = _cell_number(cell)
        if not isinstance(u, Function) or piece != 0 or cell not in range(width):
            raise NotImplementedError("expected a local Basix field read")
        E = _element(u.space)
        dof = _derived_index("compiler.ufcx.dof", u, cell)
        block_size = u.space.block_size
        if width == 1 and u in argument_rows:
            dof = argument_rows[u]

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
                coordinate_data[cell, dof, geometry.index(u)]
                if u in geometry
                else inputs[u][cell, dof, component_number]
            )
            return index_sum(data * value, (dof,))

        return arrayed((*u.value_shape, *(points.value_shape[0],) * order), component)

    # Replace whole field reads before visiting their Function operands.
    def visit(visit, node):
        replacement = tabulate(node)
        return replacement if replacement is not node else _map_fields(node, visit, _lowered(node))

    return simplify(_memoized(visit)(f))


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
