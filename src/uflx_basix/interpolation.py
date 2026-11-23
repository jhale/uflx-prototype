"""Symbolic interpolation using library basis data and core mapping operations."""

import numpy as np

from uflx.calculus import integral
from uflx.geometry import domain_map, pull_back
from uflx.nodes import AbstractExpression, EuclideanSpace
from uflx.tensors import (
    component_tensor,
    index,
    index_sum,
    matrix_product,
    tensor,
)

from .nodes import BasixBasis, BasixCell, BasixSpace, table


def interpolate(f: AbstractExpression, V: BasixSpace) -> AbstractExpression:
    """Interpolate f on the reference cell using the target element's value DOFs."""
    if not isinstance(V, BasixSpace):
        raise ValueError("interpolation requires a Basix space")
    K = V.domain
    if not isinstance(K, BasixCell):
        raise ValueError("interpolation requires a target space on a Basix reference cell")
    if f.value_shape != V.value_shape or f.scalar_type != V.scalar_type:
        raise ValueError("input value shape and scalar type must match the target space")
    if f.argument_spaces:
        raise ValueError("function inputs must be empty")
    if f.domain not in (None, K):
        raise ValueError("input domain must match the target cell")
    E = V.element(0)
    if E.interpolation_nderivs:
        raise NotImplementedError("interpolation with derivative DOFs is not supported")

    def sample(point):
        position = table(point)
        return integral(pull_back(f, domain_map(EuclideanSpace(0), position, K)), EuclideanSpace(0))

    samples = tuple(sample(point) for point in E.points)
    basis = BasixBasis(V, 0)

    # Basix orders matrix columns by value component, then by sampling point.
    axes = tuple(index(("interpolate", f, basis, "value", k)) for k in range(len(V.value_shape)))
    a = index(("interpolate", f, basis, "dof"))
    if V.block_shape:
        values = tensor(tuple(value[axes] for value in samples))
        phi = basis[a]
    else:
        values = tensor(
            tuple(value[c] for c in np.ndindex(tuple(E.value_shape)) for value in samples)
        )
        phi = basis[(a, *axes)]
    dofs = matrix_product(table(E.interpolation_matrix), values)
    return component_tensor(index_sum(dofs[a] * phi, (a,)), axes)
