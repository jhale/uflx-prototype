"""Numerical data supplied by the Basix extension."""

from functools import cache
from math import prod

import jax
import jax.numpy as jnp
import numpy as np

from uflx.nodes import AbstractExpression
from uflx_basix.nodes import ReferenceSimplex

from ._context import lookup

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


@cache
def cell_quadrature(cell):
    """Return reference points and weights for numerical integration."""
    if cell.dimension == 0:
        return np.empty((1, 0)), np.ones(1)
    if is_basix_cell(cell):
        cell_type = cell.cell_type
    elif isinstance(cell, ReferenceSimplex) and 1 <= cell.dimension <= 3:
        if basix is None:
            raise ModuleNotFoundError("simplex quadrature requires fenics-basix", name="basix")
        simplices = (basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron)
        cell_type = simplices[cell.dimension - 1]
    else:
        raise NotImplementedError(f"no reference quadrature for {cell}")
    return basix.make_quadrature(cell_type, 12)


@cache
def _tabulator(element, orders):
    """Basix provides exact derivative tables even at traced evaluation points."""

    def host(point):
        points = np.asarray(point, dtype=element.dtype)
        batch = points.shape[:-1]
        table = element.tabulate(sum(orders), points.reshape(prod(batch), len(orders)))
        values = table[basix.index(*orders) if orders else 0]
        return values.reshape(*batch, element.dim, element.value_size).astype(
            point.dtype, copy=False
        )

    @jax.custom_jvp
    def tabulate(point):
        if not isinstance(point, jax.core.Tracer):
            return jnp.asarray(host(point))
        shape = jax.ShapeDtypeStruct((element.dim, element.value_size), point.dtype)
        return jax.pure_callback(host, shape, point, vmap_method="broadcast_all")

    @tabulate.defjvp
    def derivative(primals, tangents):
        (point,), (direction,) = primals, tangents
        value = tabulate(point)
        tangent = jnp.zeros_like(value)
        for axis in range(len(orders)):
            partial = (*orders[:axis], orders[axis] + 1, *orders[axis + 1 :])
            tangent = tangent + _tabulator(element, partial)(point) * direction[axis]
        return value, tangent

    return tabulate


def tabulate(element, point):
    point = jnp.asarray(point, dtype=float)
    return _tabulator(element, (0,) * len(point))(point)


def field_value(node, bindings, site, value):
    space = node.space
    topology = space.domain
    coefficients = lookup(bindings, node, f"coefficients:{node.label}")
    if isinstance(coefficients, AbstractExpression):
        return value(coefficients)
    if callable(coefficients):
        # A form slot may supply a field directly on the reference cell.
        return jnp.asarray(coefficients(site.points[topology]))
    slot = site.slot(topology)
    basis = tabulate(space.element(slot), site.points[topology])

    local = jnp.asarray(coefficients)[np.asarray(space.cell_dofs[slot])]
    if space.block_shape:
        return (local.reshape(-1, space.block_size).T @ basis[:, 0]).reshape(space.value_shape)
    return (local @ basis).reshape(space.value_shape)


def basis_value(node, site):
    space = node.space
    if site.slot(space.domain) != node.cell_slot:
        return np.zeros(node.value_shape)
    basis = tabulate(space.element(node.cell_slot), site.points[space.domain])
    return basis.reshape(node.value_shape)


@cache
def vertex_element(cell_type):
    return basix.create_element(
        basix.ElementFamily.P, cell_type, 1, basix.LagrangeVariant.equispaced
    )


def vertex_value(topology, vertices, site):
    slot = site.slot(topology)
    cell = topology.cell_types[slot]
    point = site.points[topology]
    if is_basix_cell(cell) and cell.dimension:
        weights = tabulate(vertex_element(cell.cell_type), point)[:, 0]
    elif cell.num_vertices == cell.dimension + 1:
        weights = jnp.concatenate((jnp.atleast_1d(1 - jnp.sum(point)), point))
    else:
        bits = np.array(
            [[(v >> d) & 1 for d in range(cell.dimension)] for v in range(cell.num_vertices)]
        )
        weights = np.ones(cell.num_vertices)
        for axis in range(cell.dimension):
            x = point[axis]
            factors = 1 - bits[:, axis] + (2 * bits[:, axis] - 1) * x
            weights = weights * factors
    return weights @ np.asarray(vertices)[list(topology.cells[slot])]
