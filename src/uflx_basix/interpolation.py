"""Symbolic interpolation using library basis data and core mapping operations."""

from math import prod

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


def interpolate(
    f: AbstractExpression,
    V: BasixSpace,
    *,
    sampling_rule: tuple[np.ndarray, np.ndarray] | None = None,
) -> AbstractExpression:
    """Interpolate f on a Basix reference cell and return a UFLx expression.

    The optional (points, matrix) pair replaces the element's value-sampling
    rule. Matrix columns run over components first, then points.
    """
    if not isinstance(V, BasixSpace):
        raise ValueError("interpolation requires a Basix space")
    K = V.domain
    if not isinstance(K, BasixCell):
        raise ValueError("interpolation requires a target space on a Basix reference cell")
    if f.value_shape != V.value_shape or f.scalar_type != V.scalar_type:
        raise ValueError("input value shape and scalar type must match the target space")
    if f.argument_spaces:
        raise ValueError("function inputs must be empty")
    basis = BasixBasis(V, 0)
    E = V.element(0)
    if f.domain not in (None, K):
        raise ValueError("input domain must match the target cell")
    if E.interpolation_nderivs:
        raise NotImplementedError("interpolation with derivative DOFs is not supported")
    points, matrix = (E.points, E.interpolation_matrix) if sampling_rule is None else sampling_rule
    points, matrix = np.asarray(points), np.asarray(matrix)
    components = prod(E.value_shape)
    if points.ndim != 2 or points.shape[1] != K.dimension:
        raise ValueError("sampling points must have shape (num_points, cell_dimension)")
    if matrix.shape != (E.dim, components * len(points)):
        raise ValueError("sampling matrix has the wrong shape")
    if not np.all(np.isfinite(points)) or not np.all(np.isfinite(matrix)):
        raise ValueError("sampling data must be finite")

    def sample(point):
        position = table(point)
        return integral(pull_back(f, domain_map(EuclideanSpace(0), position, K)), EuclideanSpace(0))

    samples = tuple(sample(point) for point in points)

    # Interpolation is the sampling matrix applied to values, followed by a
    # contraction with the target basis. Blocked scalar elements share the rule.
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
    dofs = matrix_product(table(matrix), values)
    return component_tensor(index_sum(dofs[a] * phi, (a,)), axes)
