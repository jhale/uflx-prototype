"""Finite-element tabulation for the JAX backend."""

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


def _basis_field(space, dof):
    def read(site, slots):
        slot = site.slot(space.domain)
        local = (jnp.asarray(space.cell_dofs[slot]) == dof).astype(dtype(space))
        return _local_value(space, slot, site, local)

    return BoundExpression(space, read, needs_eager_samples=False)


def local_basis(slot):
    """Return the dimension and indexed basis of a Basix form slot."""
    members = slot if isinstance(slot, tuple) else (slot,)
    spaces = tuple(basix_space(s) for s in members)
    if None in spaces:
        return None
    offsets = np.cumsum([0, *(s.num_dofs for s in spaces)])

    def basis(dof):
        fields = tuple(
            _basis_field(space, dof - start)
            for space, start in zip(spaces, offsets[:-1], strict=True)
        )
        return fields if isinstance(slot, tuple) else fields[0]

    return offsets[-1], basis


def is_basix_basis(node):
    return basix is not None and isinstance(node, BasixBasis)


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
    basis = tabulate(space.element(slot), site.points[space.domain])
    if space.block_shape:
        return jnp.tensordot(basis[:, 0], local.reshape(-1, *space.block_shape), axes=1)
    return (local @ basis).reshape(space.value_shape)


def basis_value(node, site):
    if site.slot(node.domain) != node.cell_slot:
        return np.zeros(node.value_shape)
    basis = tabulate(node.element, site.points[node.domain])
    return basis.reshape(node.value_shape)
