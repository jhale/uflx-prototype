"""Numerical meanings of spatial derivatives, field mappings and conormals."""

import numpy as np

from uflx.nodes import AbstractTopology, CellMap, Mesh, SideMap

from ._geometry import (
    chart_of,
    jacobian,
    reference_point,
    side_reference,
    tangent_projector,
    transformed,
)
from ._tensors import size


def gradient(operand, domain, frame, arguments):
    ctx = frame.context
    point = np.asarray(ctx.points[domain])
    if point.size == 0:
        shape = tuple(size(n, frame) for n in operand.value_shape)
        return np.empty((*shape, 0), dtype=complex if operand.scalar_type == "complex" else float)

    def field(p):
        return ctx.at({domain: p}, lambda: np.asarray(frame.value(operand, arguments)))

    result = jacobian(field, point, ctx.spatial_step)
    if isinstance(domain, Mesh) and domain.dimension < len(point):
        result = result @ tangent_projector(domain, frame)
    return result


def curl(operand, domain, frame, arguments):
    g = gradient(operand, domain, frame, arguments)
    if g.shape[-1] == 2:
        return g[1, 0] - g[0, 1]
    return np.array([g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]])


def pull_back(operand, mapping, frame, arguments):
    ctx = frame.context
    if isinstance(mapping, SideMap):
        _, slot, _, _, point = side_reference(mapping, frame)
        parent = mapping.parent
        if isinstance(parent, Mesh):
            point = chart_of(parent[slot], frame)(point)
        bindings = {("piece", parent): slot, parent: point}
    elif isinstance(mapping, CellMap) and isinstance(mapping.parent, AbstractTopology):
        bindings = {
            ("piece", mapping.parent): mapping.cell_slot,
            mapping.parent: ctx.points[mapping.reference_cell],
        }
    else:
        point = np.real(np.asarray(frame.value(mapping.parameterisation)))
        if isinstance(mapping, CellMap):
            bindings = {("piece", mapping.parent): mapping.cell_slot, mapping.parent: point}
        else:
            bindings = {mapping.ambient_domain: point}
    return ctx.at(bindings, lambda: frame.value(operand, arguments))


def push_forward(operand, mesh, law, frame, arguments):
    ctx = frame.context
    slot = ctx.points.get(("piece", mesh), 0)
    chart = chart_of(mesh[slot], frame)
    ref = reference_point(chart, ctx.points[mesh], mesh.dimension, ctx.spatial_step)
    topology = mesh.topology
    field = ctx.at(
        {("piece", topology): slot, topology: ref}, lambda: frame.value(operand, arguments)
    )
    return transformed(field, law, jacobian(chart, ref, ctx.spatial_step))


def conormal(side, conormals, frame):
    slot, parent_slot, inclusion, ref, target = side_reference(side, frame)
    J = jacobian(chart_of(side.parent[parent_slot], frame), target, frame.context.spatial_step)
    normal = frame.context.at(
        {inclusion.parameter_domain: ref}, lambda: np.real(np.asarray(frame.value(conormals[slot])))
    )
    vector = np.linalg.pinv(J).T @ normal
    return vector / np.linalg.norm(vector)
