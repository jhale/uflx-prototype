# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Finite-element tabulation for the JAX backend."""

from itertools import combinations_with_replacement
from math import comb, factorial, prod

import jax
import jax.numpy as jnp
import numpy as np

from uflx.nodes import AbstractExpression, AbstractTopology, ConjugateSpace

from ._context import BoundExpression, lookup
from ._tensors import dtype

try:
    import basix
except ModuleNotFoundError as error:
    if error.name != "basix":
        raise
    basix = None
else:
    from uflx_basix.nodes import BasixBasis, BasixSpace

    from ._tabulation import tabulate


def basix_space(space):
    if isinstance(space, ConjugateSpace):
        space = space.space
    return space if basix is not None and isinstance(space, BasixSpace) else None


def _basis_field(space, dof):
    def read(site, slots):
        slot = site.slot(space.domain)
        local = (jnp.asarray(space.cell_dofs[slot]) == dof).astype(dtype(space))
        return _local_value(space, slot, site, local)

    return BoundExpression(space, read, needs_eager_samples=False)


def local_basis(slot):
    """Return the dimension and indexed basis of a Basix form slot."""
    members, structure = jax.tree.flatten(slot)
    spaces = tuple(basix_space(s) for s in members)
    if None in spaces:
        return None
    offsets = np.cumsum([0, *(s.num_dofs for s in spaces)])

    def basis(dof):
        fields = tuple(
            _basis_field(space, dof - start)
            for space, start in zip(spaces, offsets[:-1], strict=True)
        )
        return jax.tree.unflatten(structure, fields)

    return offsets[-1], basis


def is_basix_basis(node):
    return basix is not None and isinstance(node, BasixBasis)


def tabulation_space(selector):
    return selector.space if is_basix_basis(selector) else basix_space(selector)


def quadrature_tables(integral, cell, bindings, npoints):
    """Collect the Basix tables declared for this integral, in point-major order."""
    tables = {}
    for key, data in bindings.items():
        if not (isinstance(key, tuple) and len(key) == 2 and key[0] == integral):
            continue
        selector = key[1]
        space = tabulation_space(selector)
        if space is None:
            continue
        domain = space.domain
        cells = domain.cell_types if isinstance(domain, AbstractTopology) else (domain,)
        slots = [i for i, K in enumerate(cells) if K == cell]
        if is_basix_basis(selector):
            slot = 0 if space.cell is not None else selector.cell_slot
            slots = [slot] if slot in slots else []
        if not slots:
            raise ValueError("the tabulated space does not use the integration cell")
        if len({space.element_data[i] for i in slots}) != 1:
            raise ValueError(
                "select a BasixBasis(space, slot) for different elements on the same cell"
            )
        element = space.element(slots[0])
        table = jnp.asarray(data)
        if (
            table.ndim != 4
            or table.shape[0] < 1
            or table.shape[1:] != (npoints, element.dim, element.value_size)
            or not jnp.issubdtype(table.dtype, jnp.floating)
        ):
            raise ValueError(
                f"expected a floating Basix tabulation with shape "
                f"(derivatives, {npoints}, {element.dim}, {element.value_size})"
            )
        for slot in slots:
            if (space, slot) in tables:
                raise ValueError("duplicate tabulation for the same space and cell slot")
            tables[space, slot] = jnp.swapaxes(table, 0, 1)
    return tables


def _basis_at(space, slot, domain, site):
    point = site.points[domain]
    tabulation = site.tabulations.get(domain)
    rows = tabulation.rows.get((space, slot)) if tabulation else None
    if rows is None:
        return tabulate(space.element(slot), point)
    order = tabulation.derivative_order
    if not order:
        return rows[0]
    dim = point.shape[0]
    if rows.shape[0] < comb(dim + order, order):
        raise ValueError(f"runtime basis tabulation needs derivatives through order {order}")
    # Spatial AD evaluates this Taylor polynomial at its center. Each partial is
    # supplied by the caller, including mixed derivatives in Basix ordering.
    delta = point - tabulation.center
    result = rows[0] + jnp.tensordot(delta, rows[1 : dim + 1], axes=1)
    for degree in range(2, order + 1):
        for axes in combinations_with_replacement(range(dim), degree):
            powers = tuple(axes.count(i) for i in range(dim))
            scale = prod(delta[i] for i in axes) / prod(factorial(n) for n in powers)
            result = result + scale * rows[basix.index(*powers)]
    return result


def field_value(node, space, bindings, site, value):
    topology = space.domain
    coefficients = lookup(bindings, node, f"coefficients:{node.label}")
    if isinstance(coefficients, AbstractExpression):
        return value(coefficients)
    if callable(coefficients):
        # A form slot may supply a field directly on the reference cell.
        return jnp.asarray(coefficients(site.points[topology]))
    slot = site.slot(topology)
    values = jnp.asarray(coefficients)
    dofs = np.asarray(space.cell_dofs[slot])
    if space.block_shape:
        flat_shape = (space.num_dofs,)
        nodal_shape = (space.num_dofs // space.block_size, *space.block_shape)
        if values.shape not in (flat_shape, nodal_shape):
            raise ValueError(
                f"expected coefficient shape {flat_shape} or {nodal_shape} (got {values.shape})"
            )
        values = values.reshape(-1)
    return _local_value(space, slot, site, values[dofs])


def _local_value(space, slot, site, local):
    basis = _basis_at(space, slot, space.domain, site)
    if space.block_shape:
        return jnp.tensordot(basis[:, 0], local.reshape(-1, *space.block_shape), axes=1)
    return (local @ basis).reshape(space.value_shape)


def basis_value(node, site):
    if site.slot(node.domain) != node.cell_slot:
        return np.zeros(node.value_shape)
    slot = 0 if node.space.cell is not None else node.cell_slot
    basis = _basis_at(node.space, slot, node.domain, site)
    return basis.reshape(node.value_shape)
