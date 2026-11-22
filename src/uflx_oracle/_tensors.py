"""Tensor operations, runtime extents and index assignments."""

from itertools import product

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractExtent, GatheredIndex, Index, IndexSum

from ._forms import members


def size(extent, frame):
    if isinstance(extent, tuple):
        return tuple(size(n, frame) for n in extent)
    return frame.input(extent, extent.label) if isinstance(extent, AbstractExtent) else extent


def index_value(index, indices):
    if isinstance(index, Index):
        return indices[index]
    if isinstance(index, GatheredIndex):
        value = index.table.values
        for i in index.indices:
            key = index_value(i, indices)
            if isinstance(key, jax.core.Tracer):
                value = jnp.asarray(value)
            value = value[key]
        if getattr(value, "ndim", 0):
            return tuple(value)
        return value
    return index


def flat(entries):
    return tuple(axis for entry in entries for axis in members(entry))


def bound_tensor(node, frame, arguments):
    operand = node.expression
    dimensions = dict(operand.free_indices)
    extents = [size(dimensions[i], frame) for i in node.indices]
    ranges = [product(*map(range, n)) if isinstance(n, tuple) else range(n) for n in extents]
    dtype = {"boolean": bool, "complex": complex, "real": float}[node.scalar_type]
    summed = isinstance(node, IndexSum)
    total = dtype(0)
    entries = []
    for combo in product(*ranges):
        bound = frame.bind_indices(dict(zip(node.indices, combo, strict=True)))
        value = bound.value(operand, arguments)
        if summed:
            total = total + value
        else:
            entries.append(jnp.real(value) if node.scalar_type == "real" else value)
    if summed:
        return total
    if not entries:
        return jnp.empty(flat(extents), dtype=dtype)
    return jnp.asarray(entries, dtype=dtype).reshape(flat(extents))


def solve(matrix, rhs):
    """Solve a full-column-rank system, in the least-squares sense if rectangular."""
    if matrix.shape[0] != matrix.shape[1]:
        Q, R = jnp.linalg.qr(matrix)
        return jnp.linalg.solve(R, Q.T @ rhs)
    return jnp.linalg.solve(matrix, rhs)


def inverse(matrix):
    return solve(matrix, jnp.eye(matrix.shape[0], dtype=matrix.dtype))


@jax.custom_jvp
def determinant(matrix):
    return jnp.linalg.det(matrix)


@determinant.defjvp
def _determinant_jvp(primals, tangents):
    (matrix,), (direction,) = primals, tangents
    value = determinant(matrix)
    tangent = jnp.zeros_like(value)
    # Cofactors depend only on the primal, so JAX can transpose this linear map.
    for i in range(len(matrix)):
        rows = jnp.delete(matrix, i, axis=0)
        for j in range(len(matrix)):
            minor = jnp.delete(rows, j, axis=1)
            minor_det = minor[0, 0] if minor.shape == (1, 1) else determinant(minor)
            cofactor = (-1) ** (i + j) * minor_det
            tangent = tangent + cofactor * direction[i, j]
    return value, tangent
