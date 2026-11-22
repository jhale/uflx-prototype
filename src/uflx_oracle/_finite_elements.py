"""Numerical data supplied by the Basix extension."""

from functools import cache

import numpy as np

from . import _dual as ad
from ._context import call_at

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


def tabulate(element, point, orders=None):
    """Read exact partial derivatives of a basis from the Basix derivative table."""
    if orders is None:
        orders = (0,) * np.size(ad.primal(point))
    if isinstance(point, ad.Dual):
        base = tabulate(element, point.value, orders)
        tangent = np.zeros_like(ad.primal(base))
        for axis in range(len(orders)):
            partial = (*orders[:axis], orders[axis] + 1, *orders[axis + 1 :])
            tangent = tangent + tabulate(element, point.value, partial) * point.tangent[axis]
        return ad.Dual(point.tag, base, tangent)
    table = element.tabulate(sum(orders), np.asarray([point], dtype=float))
    return table[basix.index(*orders) if orders else 0, 0]


def field_value(node, frame):
    space = node.space
    points = frame.points
    topology = space.domain
    slot = points.get(("piece", topology), 0)
    coefficients = frame.input(node, f"coefficients:{node.label}")
    if callable(coefficients):
        # A form slot may supply a field directly on the reference cell.
        return call_at(points, ad.call_field, coefficients, points[topology])
    basis = tabulate(space.element(slot), points[topology])
    local = ad.asarray(coefficients)[list(space.cell_dofs[slot])]
    value = (
        local.reshape(-1, space.block_size).T @ basis[:, 0] if space.block_shape else local @ basis
    )
    return value.reshape(space.value_shape)


def basis_value(node, points):
    space = node.space
    if points.get(("piece", space.domain), 0) != node.cell_slot:
        return np.zeros(node.value_shape)
    basis = tabulate(space.element(node.cell_slot), points[space.domain])
    return basis.reshape(node.value_shape)


@cache
def vertex_element(cell_type):
    return basix.create_element(
        basix.ElementFamily.P, cell_type, 1, basix.LagrangeVariant.equispaced
    )


def vertex_value(topology, vertices, points):
    slot = points.get(("piece", topology), 0)
    cell = topology.cell_types[slot]
    point = points[topology]
    if is_basix_cell(cell) and cell.dimension:
        weights = tabulate(vertex_element(cell.cell_type), point)[:, 0]
    elif cell.num_vertices == cell.dimension + 1:
        weights = ad.stack([1 - np.sum(point), *(point[i] for i in range(cell.dimension))])
    else:
        bits = np.array(
            [[(v >> d) & 1 for d in range(cell.dimension)] for v in range(cell.num_vertices)]
        )
        weights = np.ones(cell.num_vertices)
        for axis in range(cell.dimension):
            x = point[axis]
            weights = weights * ((1 - bits[:, axis]) * (1 - x) + bits[:, axis] * x)
    return weights @ np.asarray(vertices)[list(topology.cells[slot])]
