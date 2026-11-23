"""Tensor operations, runtime extents and index assignments."""

import jax
import jax.numpy as jnp
import numpy as np

from uflx.nodes import AbstractExtent, ComponentTensor, GatheredIndex, Index, Product

from ._context import lookup


def size(extent, bindings):
    return lookup(bindings, extent) if isinstance(extent, AbstractExtent) else extent


def axes(node, bindings):
    """Leading axes in the numerical array. An explicitly selected index has no axis."""
    return tuple(i for i, _ in node.free_indices if i not in bindings)


def shape(node, bindings):
    free = (n for i, n in node.free_indices if i not in bindings)
    return tuple(size(n, bindings) for n in (*free, *node.value_shape))


def dtype(node):
    return {"boolean": bool, "complex": complex, "real": float}[node.scalar_type]


def align(data, source, target, padding=0):
    """Put free axes in target order and insert singleton axes for broadcasting."""
    if source == target and not padding:
        return data
    data = jnp.asarray(data)
    present = [source.index(i) for i in target if i in source]
    data = jnp.transpose(data, (*present, *range(len(source), data.ndim)))
    missing = tuple(k for k, i in enumerate(target) if i not in source)
    return jnp.expand_dims(data, (*missing, *range(len(target), len(target) + padding)))


def indexed(node, array, bindings):
    """Gather from value axes, broadcasting index tables over the free axes."""
    order = axes(node, bindings)
    source = axes(node.expression, bindings)
    if not source and node.indices == order:
        return array
    extents = dict(node.free_indices)

    def entry(i):
        if isinstance(i, GatheredIndex):
            table = i.table
            if table in bindings:
                values = jnp.asarray(bindings[table])
                expected = tuple(size(n, bindings) for n in table.shape)
                if values.shape != expected or not jnp.issubdtype(values.dtype, jnp.integer):
                    raise ValueError(f"expected an integer index table of shape {expected}")
            else:
                values = jnp.asarray(table.values)
            return values[tuple(map(entry, i.indices))]
        if isinstance(i, Index):
            if i in bindings:
                return bindings[i]
            values = jnp.arange(size(extents[i], bindings))
            return align(values, (i,), order)
        return i

    key = tuple(map(entry, (*source, *node.indices)))
    return jnp.asarray(array)[key]


def bind_indices(node, bindings, arguments, evaluate):
    """A tensor binder reorders axes or sums them out of its scalar body."""
    body = node.expression
    bound = node.indices
    # A binder shadows an index selected by its caller.
    local = {k: v for k, v in bindings.items() if k not in bound}
    output = axes(node, bindings)
    order = axes(body, local)
    extents = dict(body.free_indices)
    if any(size(extents[i], local) == 0 for i in bound):
        return jnp.zeros(shape(node, bindings), dtype=dtype(node))

    if isinstance(node, ComponentTensor):
        data = align(evaluate(body, local, arguments), order, output + bound)
        if node.scalar_type == "real":
            data = jnp.real(data)
        return jnp.asarray(data, dtype=dtype(node))

    # Index sums are real or complex even when a runtime array stores integers.
    def numerical(expr, slots):
        data = evaluate(expr, local, slots)
        return jnp.asarray(data, dtype=jnp.result_type(data, dtype(expr)(0)))

    if isinstance(body, Product):

        def factors(expr, slots):
            if isinstance(expr, Product):
                n = len(expr.first.argument_spaces)
                yield from factors(expr.first, slots[:n])
                yield from factors(expr.second, slots[n:])
            else:
                yield numerical(expr, slots)
                yield [order.index(i) for i in axes(expr, local)]

        return jnp.einsum(*factors(body, arguments), [order.index(i) for i in output])

    data = numerical(body, arguments)
    return jnp.sum(data, axis=tuple(order.index(i) for i in bound))


def pointwise(node, bindings, evaluate, *, batch=False):
    """Evaluate one scalar-index rule over the Cartesian product of its free axes."""
    order = axes(node, bindings)
    result_shape = shape(node, bindings)
    extents = result_shape[: len(order)]

    def entry(*key):
        return evaluate(dict(zip(order, key, strict=True)))

    if batch:
        # Separate maps let JAX keep each operand outside the axes it does not use.
        mapped = entry
        for axis in reversed(range(len(order))):
            in_axes = tuple(0 if i == axis else None for i in range(len(order)))
            mapped = jax.vmap(mapped, in_axes=in_axes)
        return mapped(*(jnp.arange(n) for n in extents))
    return jnp.stack([entry(*key) for key in np.ndindex(extents)]).reshape(result_shape)


def solve(matrix, rhs):
    """Solve a full-column-rank system, in the least-squares sense if rectangular."""
    if matrix.shape[0] != matrix.shape[1]:
        Q, R = jnp.linalg.qr(matrix)
        return jnp.linalg.solve(R, Q.T @ rhs)
    return jnp.linalg.solve(matrix, rhs)


def _small_inverse(matrix):
    """Invert a 2x2 or 3x3 matrix with pivoted row operations unrolled by JAX."""
    n = matrix.shape[0]
    matrix = matrix.astype(jnp.result_type(matrix, 0.0))
    rows = list(jnp.concatenate((matrix, jnp.eye(n, dtype=matrix.dtype)), axis=1))

    # Eliminate below each pivot in [A | I]. Pivoting avoids division by a
    # small leading entry and does not require forming a determinant.
    for k in range(n - 1):
        for i in range(k + 1, n):
            pivot, row = rows[k], rows[i]
            swap = jnp.abs(row[k]) > jnp.abs(pivot[k])
            rows[k] = jnp.where(swap, row, pivot)
            rows[i] = jnp.where(swap, pivot, row)
        for i in range(k + 1, n):
            rows[i] = rows[i] - (rows[i][k] / rows[k][k]) * rows[k]

    result = [row[n:] for row in rows]
    for k in reversed(range(n)):
        for j in range(k + 1, n):
            result[k] = result[k] - rows[k][j] * result[j]
        result[k] = result[k] / rows[k][k]
    return jnp.stack(result)


def inverse(matrix):
    if matrix.shape in ((2, 2), (3, 3)):
        return _small_inverse(matrix)
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
