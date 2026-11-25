# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Finite-element tabulation for the JAX backend."""

from itertools import combinations_with_replacement, product
from math import comb, factorial, prod

import jax
import jax.numpy as jnp
import numpy as np

from uflx.nodes import (
    AbstractExpression,
    AbstractTopology,
    CellMap,
    ConjugateSpace,
    Function,
    Gradient,
    PullBack,
)

from ._context import BoundExpression, PointTabulation, lookup
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

    return BoundExpression(space, read, needs_eager_samples=False, is_basis_field=True)


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


def vertex_p1_simplex(element):
    """Whether scalar DOFs are values at the reference simplex vertices, in order."""
    return (
        element.family == basix.ElementFamily.P
        and element.degree == 1
        and element.cell_type
        in (basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron)
        and element.interpolation_is_identity
        and not element.dof_ordering
        and np.array_equal(element.points, basix.geometry(element.cell_type))
    )


def tabulated_gradient(gradient, bindings, site, read):
    """Contract native reference derivative rows with the usual field DOFs or form slot."""
    domain = gradient.domain
    field = gradient
    order = 0
    while isinstance(field, Gradient) and field.domain == domain:
        field = field.expression
        order += 1
    slot = site.slot(domain)
    leaf = field
    if (
        isinstance(field, PullBack)
        and isinstance(field.map, CellMap)
        and isinstance(field.map.parent, AbstractTopology)
        and field.map.reference_cell == domain
    ):
        slot, leaf = field.map.cell_slot, field.expression
    if isinstance(leaf, Function) and basix_space(leaf.space) is not None:
        binding = lookup(bindings, leaf, f"coefficients:{leaf.label}")
        if (
            callable(binding)
            or isinstance(binding, AbstractExpression)
            or (isinstance(binding, BoundExpression) and not binding.is_basis_field)
        ):
            raise ValueError(
                "Spatial derivative of a supplied field requires an explicit derivative"
            )
    elif is_basix_basis(leaf) and slot != leaf.cell_slot:
        return jnp.zeros(gradient.value_shape, dtype=dtype(leaf))
    elif not is_basix_basis(leaf):
        raise ValueError(
            f"Spatial derivative of {type(field).__name__} requires lowering before tracing"
        )

    element_slot = 0 if leaf.space.cell is not None else slot
    data = leaf.space.element_data[element_slot]
    point = site.points[domain]
    dim = point.shape[0]
    table = site.tabulations.get(domain)
    rows = table.rows.get(data) if table is not None else None
    if (
        order == 1
        and isinstance(leaf, Function)
        and leaf.space.block_shape
        and not isinstance(binding, BoundExpression)
        and (rows is None or data not in site.runtime_table_elements)
        and vertex_p1_simplex(data.native)
    ):
        # The columns of the affine Jacobian are edges from reference vertex zero.
        local = _cell_coefficients(leaf.space, element_slot, binding)
        vertices = local.reshape(dim + 1, *leaf.space.block_shape)
        vertices = vertices.astype(jnp.result_type(vertices, point if rows is None else rows))
        return jnp.moveaxis(vertices[1:] - vertices[0], 0, -1)
    if rows is None:
        # The highest derivatives of a simplex polynomial are constant, including
        # the Jacobian of a P1 simplex even when its evaluation point is dynamic.
        element = data.native
        if (
            element.polyset_type != basix.PolysetType.standard
            or element.cell_type
            not in (basix.CellType.interval, basix.CellType.triangle, basix.CellType.tetrahedron)
            or order < element.embedded_superdegree
        ):
            raise ValueError(
                "Expected a Basix derivative table at this point. "
                "Use fixed reference points or supply the input from basis_tables()"
            )
        rows = jnp.asarray(
            element.tabulate(order, np.zeros((1, dim), dtype=element.dtype))[:, 0],
            dtype=point.dtype,
        )
    if rows.shape[0] < comb(dim + order, order):
        raise ValueError(f"runtime basis tabulation needs derivatives through order {order}")
    # Basix packs mixed partials once. Gradient has one tensor axis per derivative.
    indices = [
        basix.index(*(axes.count(i) for i in range(dim)))
        for axes in product(range(dim), repeat=order)
    ]

    def component(row):
        at = site.at(
            {domain: point}, tabulations={domain: PointTabulation(point, {data: row[None]})}
        )
        return read(field, at=at)

    values = jax.vmap(component)(rows[np.asarray(indices)])
    return jnp.moveaxis(values, 0, -1).reshape((*values.shape[1:], *(dim,) * order))


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
    return _local_value(space, slot, site, _cell_coefficients(space, slot, coefficients))


def _cell_coefficients(space, slot, coefficients):
    values = jnp.asarray(coefficients)
    dofs = space.cell_dofs[slot]
    if space.block_shape:
        flat_shape = (space.num_dofs,)
        nodal_shape = (space.num_dofs // space.block_size, *space.block_shape)
        if values.shape not in (flat_shape, nodal_shape):
            raise ValueError(
                f"expected coefficient shape {flat_shape} or {nodal_shape} (got {values.shape})"
            )
        values = values.reshape(-1)
    return values[dofs[0] : dofs[-1] + 1] if dofs else values[:0]


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
