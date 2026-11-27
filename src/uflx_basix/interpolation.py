# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Symbolic interpolation using library basis data and core mapping operations."""

import numpy as np

from uflx.geometry import domain_map, pull_back, push_forward, transformation_law
from uflx.nodes import AbstractExpression, CellMap, Function, MappedSpace, Mesh, PullBack
from uflx.tensors import component_tensor, index, index_sum

from .evaluation import evaluate_at_points
from .nodes import BasixBasis, BasixSpace, table


def interpolate(
    f: AbstractExpression,
    V: BasixSpace | MappedSpace,
    *,
    cell: CellMap | None = None,
) -> AbstractExpression:
    """Interpolate into the reference components of V on one cell.

    A mapped target supplies its mesh and returns the physical interpolant.
    A Basix function on the mesh topology supplies its source transformation
    law. Pass cell= to supply the geometry for that function.
    Mesh and ambient expressions, and values restricted to the reference
    cell, supply physical components directly.

    A target bound with local_space(mesh[i], element) or V.on_cell(mesh[i]),
    or a source restricted to a physical cell, supplies the piece. A field on
    a single-cell mesh also supplies the cell. Otherwise pass cell=.
    The result lives on V.domain. For
    a topology-wide target it is zero on other slots and can be mapped to
    the physical mesh with push_forward(result, mesh, V.law).
    """
    if isinstance(V, MappedSpace):
        if not isinstance(V.reference_space, BasixSpace) or V.law != V.reference_space.law:
            raise ValueError("physical interpolation requires the Basix element mapping law")
        if cell is not None:
            raise ValueError("the mapped target specifies the whole physical mesh")
        pieces = iter(V.mesh)
        reference = interpolate(f, V.reference_space, cell=next(pieces))
        for piece in pieces:
            reference = reference + interpolate(f, V.reference_space, cell=piece)
        return push_forward(reference, V.mesh, V.law)
    if not isinstance(V, BasixSpace):
        raise ValueError("interpolation requires a Basix space")
    if f.argument_spaces:
        raise ValueError("function inputs must be empty")
    if f.scalar_type != V.scalar_type:
        raise ValueError("input and target scalar types must match")

    source_cell = None
    if isinstance(f, PullBack) and isinstance(f.map, CellMap) and isinstance(f.map.parent, Mesh):
        source_cell = f.map
    elif isinstance(f, Function) and isinstance(f.space, BasixSpace):
        source_cell = f.space.cell
    restricted = V.cell is not None or source_cell is not None
    for restriction in (V.cell, source_cell):
        if restriction is not None:
            if cell is not None and cell != restriction:
                raise ValueError("interpolation restrictions select different cells")
            cell = restriction

    if cell is None:
        mesh = f.domain
        if not isinstance(mesh, Mesh):
            raise ValueError("interpolation requires a physical mesh cell")
        if len(mesh.topology.cell_types) != 1:
            raise ValueError(
                "select a cell by restricting the target or field, or pass cell=mesh[i]"
            )
        cell = mesh[0]
    if not isinstance(cell, CellMap) or not isinstance(cell.parent, Mesh):
        raise ValueError("cell must be a physical mesh cell")
    mesh = cell.parent
    K = cell.reference_cell
    target_domain = V.domain
    if restricted:
        V = V.on_cell(cell)
    elif V.domain != mesh.topology:
        raise ValueError("the target space must be on the mesh topology")
    if isinstance(f, Function) and isinstance(f.space, BasixSpace) and f.domain == mesh.topology:
        f = push_forward(f, mesh, f.space.law)
    law = V.law
    unmapped = len(V.value_shape) - len(law.variances)
    physical_shape = V.value_shape[:unmapped] + (mesh.ambient_domain.dimension,) * len(
        law.variances
    )
    if f.value_shape != physical_shape:
        raise ValueError("input and physical target value shapes must match")

    if f.domain == mesh.ambient_domain:
        values = pull_back(f, domain_map(K, cell.parameterisation, mesh.ambient_domain))
    elif f.domain in (None, mesh):
        values = pull_back(f, cell)
    elif restricted and f.domain == K:
        values = f
    else:
        raise ValueError(
            "input domain must match the mesh, its ambient domain or the selected cell"
        )
    result = _interpolate_cell(values, V, cell)
    return pull_back(result, mesh.topology[cell.cell_slot]) if target_domain == K else result


def _interpolate_cell(f, V, cell):
    basis = BasixBasis(V, cell.cell_slot)
    E = basis.element
    if E.interpolation_nderivs:
        raise NotImplementedError("interpolation with derivative DOFs is not supported")

    # A DOF weight pairs with a field value, so its axes have the opposite variance.
    law = V.law
    dual_law = transformation_law(
        tuple("contravariant" if v == "covariant" else "covariant" for v in law.variances),
        -law.weight,
    )
    a = index(("interpolate", f, basis, "dof"))

    def axes(role, shape):
        return tuple(index(("interpolate", f, basis, role, k)) for k in range(len(shape)))

    blocks = axes("block", V.block_shape)
    physical = axes("physical", E.value_shape)
    reference = axes("reference", E.value_shape)

    # Basix columns run over points within each value component.
    weights = E.interpolation_matrix.reshape(E.dim, *E.value_shape, len(E.points))
    q = index(("interpolate", f, basis, "point"))
    weights_at_q = table(np.moveaxis(weights, -1, 0))[q]

    directions = push_forward(weights_at_q, cell.parent, dual_law)
    physical_weights = pull_back(directions, cell)

    pairing = index_sum(physical_weights[(a, *physical)] * f[(*blocks, *physical)], physical)
    samples = evaluate_at_points(pairing, E.points, q, cell=cell.reference_cell)
    dofs = index_sum(samples, (q,))

    return component_tensor(index_sum(dofs * basis[(a, *reference)], (a,)), (*blocks, *reference))
