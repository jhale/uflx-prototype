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

from uflx.nodes import AbstractExpression, ConjugateSpace

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


def _basis_field(space, dof, whole_axis):
    def read(site, slots):
        slot = site.slot(space.domain)
        # Cell DOFs are contiguous, including the interleaved block components.
        local = dof if whole_axis else dof - space.cell_dofs[slot][0]
        basis = _basis_at(space, slot, space.domain, site)
        if space.block_shape:
            column, component = divmod(local, space.block_size)
            value = basis[column, 0] * (jnp.arange(space.block_size) == component)
        else:
            value = basis[local]
        if not whole_axis:
            active = (local >= 0) & (local < len(space.cell_dofs[slot]))
            value = jnp.where(active, value, 0)
        return value.reshape(space.value_shape).astype(dtype(space))

    return BoundExpression(space, read, needs_eager_samples=False)


def local_basis(slot):
    """Return the dimension and indexed basis of a Basix form slot."""
    members, structure = jax.tree.flatten(slot)
    spaces = tuple(basix_space(s) for s in members)
    if None in spaces:
        return None
    offsets = np.cumsum([0, *(s.num_dofs for s in spaces)])

    whole_axis = len(spaces) == 1 and len(spaces[0].cell_dofs) == 1

    def basis(dof):
        fields = tuple(
            _basis_field(space, dof - start, whole_axis)
            for space, start in zip(spaces, offsets[:-1], strict=True)
        )
        return jax.tree.unflatten(structure, fields)

    return offsets[-1], basis


def is_basix_basis(node):
    return basix is not None and isinstance(node, BasixBasis)


def _basis_at(space, slot, domain, site):
    point = site.points[domain]
    tabulation = site.tabulations.get(domain)
    rows = tabulation.rows.get(space.element_data[slot]) if tabulation else None
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
