"""Symbolic interpolation using library basis data and core mapping operations."""

from math import prod

import basix
import numpy as np

from uflx_spec_interpreter import (
    EuclideanSpace,
    cell_map,
    component_tensor,
    determinant,
    domain_map,
    index,
    index_sum,
    integral,
    inverse,
    jacobian,
    matrix_product,
    num,
    pull_back,
    push_forward,
    tensor,
)
from uflx_spec_interpreter._src.nodes import AbstractExpression, CellMap, Mesh

from .basix_nodes import BasixBasis, BasixCell, BasixSpace, table
from .basix_spaces import basix_space


def interpolate(
    f: AbstractExpression,
    V: BasixSpace,
    *,
    cell: CellMap | None = None,
    sampling_rule: tuple[np.ndarray, np.ndarray] | None = None,
) -> AbstractExpression:
    """Interpolate using symbolic sampling, basis reconstruction and push-forward.

    Args:
        f: A field on the target cell, or a domain-free value.
        V: A Basix space on a reference cell, or on the topology of cell.parent.
        cell: Optional physical cell map. If omitted for a mesh field, interpolate
            on every local cell. An explicit cell restricts the result to that
            cell, with zero on other cells. Maps must be invertible and between
            equal dimensions. The result is a field on the physical mesh.
        sampling_rule: Optional (points, matrix) replacing Basix's value sampling.
            Matrix columns are component-major, then point, for the unblocked
            element. The rule must be dual to the target basis. Its quadrature
            defines the interpolation, including for nonpolynomial inputs.

    Returns:
        A core expression retaining the symbolic dependencies of f.

    Raises:
        ValueError: If the field or sampling rule is incompatible with V.
        NotImplementedError: For unsupported cells, maps or derivative DOFs.
    """
    if not isinstance(V, BasixSpace):
        raise ValueError("interpolation requires a Basix space")
    if V.family == basix.ElementFamily.Hermite:
        raise NotImplementedError("interpolation with derivative DOFs is not supported")
    if f.value_shape != V.value_shape or f.scalar_type != V.scalar_type:
        raise ValueError("input value shape and scalar type must match the target space")
    if f.argument_spaces or V.argument_spaces:
        raise ValueError("function inputs must be empty")
    if cell is None and isinstance(f.domain, Mesh):
        fields = [
            interpolate(f, V, cell=cell_map(f.domain, j), sampling_rule=sampling_rule)
            for j in range(len(f.domain.topology.cell_types))
        ]
        return sum(fields[1:], fields[0])
    if cell is None:
        if not isinstance(V.domain, BasixCell):
            raise ValueError("a topology target requires a physical mesh field or cell map")
        return _sample_and_reconstruct(f, BasixBasis(V, 0), sampling_rule)

    if not isinstance(cell, CellMap) or not isinstance(cell.parent, Mesh):
        raise ValueError("physical interpolation requires a mesh cell map")
    D, K = cell.parent, cell.reference_cell
    if K.dimension != D.ambient_domain.dimension:
        raise NotImplementedError("physical interpolation requires equal dimensions")
    if V.domain not in (K, D.topology):
        raise ValueError("target space must be on the cell's reference cell or topology")
    if f.domain not in (None, D, D.ambient_domain):
        raise ValueError("input must be on the physical mesh or its ambient domain")
    topology_space = (
        V
        if V.domain == D.topology
        else basix_space(
            V.label,
            D.topology,
            V.family,
            V.degree,
            V.discontinuous,
            V.scalar_type,
            shape=V.block_shape,
        )
    )
    source_map = (
        domain_map(K, cell.parameterisation, D.ambient_domain)
        if f.domain == D.ambient_domain
        else cell
    )
    reference_value = _inverse_transform(pull_back(f, source_map), V.law, jacobian(cell))
    interpolant = _sample_and_reconstruct(
        reference_value, BasixBasis(topology_space, cell.cell_slot), sampling_rule
    )
    return push_forward(interpolant, D, V.law)


def _sample_and_reconstruct(f, basis, sampling_rule):
    V, j = basis.space, basis.cell_slot
    K = V.domain if isinstance(V.domain, BasixCell) else V.domain.cell_types[j]
    E = V.element(j)
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
        position = tensor(tuple(num(float(x)) for x in point))
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


def _inverse_transform(value, law, J):
    # J^T on covariant axes, J^{-1} on contravariant axes, then det(J)^weight.
    axes = tuple(
        index(("inverse_transform", value, law, J, "value", k))
        for k in range(len(value.value_shape))
    )
    contracted = tuple(
        index(("inverse_transform", value, law, J, "sum", k)) for k in range(len(law.variances))
    )
    leading = len(axes) - len(contracted)
    expression = value[(*axes[:leading], *contracted)]
    for variance, out, inner in zip(law.variances, axes[leading:], contracted, strict=True):
        expression = expression * (
            J[inner, out] if variance == "covariant" else inverse(J)[out, inner]
        )
    expression = index_sum(expression, contracted)
    if law.weight:
        expression = expression * determinant(J) ** num(law.weight)
    return component_tensor(expression, axes)
