"""Runtime extents, index assignments and enumeration of bound tensor axes."""

from itertools import product

import numpy as np

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
            value = value[index_value(i, indices)]
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
    result = dtype(0) if summed else np.empty(flat(extents), dtype=dtype)
    for combo in product(*ranges):
        bound = frame.bind_indices(dict(zip(node.indices, combo, strict=True)))
        value = bound.value(operand, arguments)
        if summed:
            result += value
        else:
            result[flat(combo)] = value if node.scalar_type != "real" else np.real(value)
    return result
