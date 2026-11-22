"""Numerical data supplied by the Basix extension."""

from functools import cache
from itertools import product
from math import comb, prod

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
def _tabulator(element, dimension, order):
    """Tabulate all partial derivatives through the given order at a reference point."""
    rows = comb(dimension + order, order)
    table_shape = (rows, element.dim, element.value_size)

    def host(point):
        points = np.asarray(point, dtype=element.dtype)
        batch = points.shape[:-1]
        table = element.tabulate(order, points.reshape(prod(batch), dimension))
        return table.swapaxes(0, 1).reshape(*batch, *table_shape).astype(point.dtype, copy=False)

    @jax.custom_jvp
    def tabulate(point):
        if not isinstance(point, jax.core.Tracer):
            return jnp.asarray(host(point))
        shape = jax.ShapeDtypeStruct(table_shape, point.dtype)
        return jax.pure_callback(host, shape, point, vmap_method="broadcast_all")

    @tabulate.defjvp
    def derivative(primals, tangents):
        (point,), (direction,) = primals, tangents
        if not dimension:
            value = tabulate(point)
            return value, jnp.zeros_like(value)
        # Each row of the current table has one partial per reference direction
        # in the next table. Basix numbers the rows by derivative multi-index.
        partials = np.empty((rows, dimension), dtype=int)
        unit = np.eye(dimension, dtype=int)
        for counts in product(range(order + 1), repeat=dimension):
            if sum(counts) <= order:
                partials[basix.index(*counts)] = [basix.index(*p) for p in counts + unit]
        higher = _tabulator(element, dimension, order + 1)(point)
        return higher[:rows], jnp.einsum("rdij,d->rij", higher[partials], direction)

    return tabulate


def tabulate(element, point):
    point = jnp.asarray(point, dtype=float)
    return _tabulator(element, len(point), 0)(point)[0]


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
