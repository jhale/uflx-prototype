"""Numerical data supplied by the Basix extension."""

from functools import cache

import numpy as np

from uflx_basix.meshes import shape_functions

try:
    import basix
except ModuleNotFoundError as error:
    if error.name != "basix":
        raise
    basix = None
else:
    from uflx_basix.basix_nodes import BasixBasis, BasixCell, BasixSpace


def is_basix_cell(cell):
    return basix is not None and isinstance(cell, BasixCell)


def is_basix_space(space):
    return basix is not None and isinstance(space, BasixSpace)


def is_basix_basis(node):
    return basix is not None and isinstance(node, BasixBasis)


def cell_quadrature(cell):
    if cell.dimension == 0:
        return [(np.empty(0), 1.0)]
    points, weights = basix.make_quadrature(cell.cell_type, 12)
    return zip(points, weights, strict=True)


def inside_cell(cell, point):
    if cell.dimension == 0:
        return True
    facets = basix.cell.topology(cell.cell_type)[-2]
    vertices = basix.geometry(cell.cell_type)
    normals = basix.cell.facet_outward_normals(cell.cell_type)
    return all(
        (point - vertices[facet[0]]) @ normal <= 1e-9
        for facet, normal in zip(facets, normals, strict=True)
    )


def field_value(node, frame):
    space = node.space
    points = frame.context.points
    topology = space.domain
    slot = points.get(("piece", topology), 0)
    coefficients = frame.input(node, f"coefficients:{node.label}")
    if callable(coefficients):
        # A form slot may supply a field directly on the reference cell.
        return coefficients(points[topology])
    basis = space.element(slot).tabulate(0, np.array([points[topology]], dtype=float))[0, 0]
    local = np.asarray(coefficients)[list(space.cell_dofs[slot])]
    value = (
        local.reshape(-1, space.block_size).T @ basis[:, 0] if space.block_shape else local @ basis
    )
    return value.reshape(space.value_shape)


def basis_value(node, points):
    space = node.space
    if points.get(("piece", space.domain), 0) != node.cell_slot:
        return np.zeros(node.value_shape)
    basis = space.element(node.cell_slot).tabulate(0, np.asarray([points[space.domain]]))[0, 0]
    return basis.reshape(node.value_shape)


@cache
def vertex_element(cell_type):
    return basix.create_element(
        basix.ElementFamily.P, cell_type, 1, basix.LagrangeVariant.equispaced
    )


def vertex_value(topology, vertices, points):
    slot = points.get(("piece", topology), 0)
    cell = topology.cell_types[slot]
    if is_basix_cell(cell) and cell.dimension:
        weights = vertex_element(cell.cell_type).tabulate(0, np.asarray([points[topology]]))[
            0, 0, :, 0
        ]
    else:
        weights = shape_functions(cell, points[topology])
    return weights @ np.asarray(vertices)[list(topology.cells[slot])]
