"""Numerical meanings of spatial derivatives, field mappings and conormals."""

import numpy as np

from uflx.nodes import AbstractTopology, CellMap, Mesh, SideMap

from . import _dual as ad
from ._geometry import (
    chart_of,
    reference_point,
    side_reference,
    tangent_projector,
    transformed,
)
from ._tensors import size


def gradient(operand, domain, frame, arguments):
    point = ad.asarray(frame.points[domain])
    if point.size == 0:
        shape = tuple(size(n, frame) for n in operand.value_shape)
        return np.empty((*shape, 0), dtype=complex if operand.scalar_type == "complex" else float)

    def field(p):
        return ad.asarray(frame.at({domain: p}).value(operand, arguments))

    result = ad.jacobian(field, point)
    if isinstance(domain, Mesh) and domain.dimension < point.size:
        result = result @ tangent_projector(domain, frame)
    return result


def curl(operand, domain, frame, arguments):
    g = gradient(operand, domain, frame, arguments)
    if g.shape[-1] == 2:
        return g[1, 0] - g[0, 1]
    return ad.stack([g[2, 1] - g[1, 2], g[0, 2] - g[2, 0], g[1, 0] - g[0, 1]])


def pull_back(operand, mapping, frame, arguments):
    if isinstance(mapping, SideMap):
        _, slot, _, _, point = side_reference(mapping, frame)
        parent = mapping.parent
        if isinstance(parent, Mesh):
            point = chart_of(parent[slot], frame)(point)
        bindings = {("piece", parent): slot, parent: point}
    elif isinstance(mapping, CellMap) and isinstance(mapping.parent, AbstractTopology):
        bindings = {
            ("piece", mapping.parent): mapping.cell_slot,
            mapping.parent: frame.points[mapping.reference_cell],
        }
    else:
        point = np.real(ad.asarray(frame.value(mapping.parameterisation)))
        if isinstance(mapping, CellMap):
            bindings = {("piece", mapping.parent): mapping.cell_slot, mapping.parent: point}
        else:
            bindings = {mapping.ambient_domain: point}
    return frame.at(bindings).value(operand, arguments)


def push_forward(operand, mesh, law, frame, arguments):
    slot = frame.points.get(("piece", mesh), 0)
    chart = chart_of(mesh[slot], frame)
    ref = reference_point(chart, frame.points[mesh], mesh.dimension)
    topology = mesh.topology
    field = frame.at({("piece", topology): slot, topology: ref}).value(operand, arguments)
    return transformed(field, law, ad.jacobian(chart, ref))


def conormal(side, conormals, frame):
    slot, parent_slot, inclusion, ref, target = side_reference(side, frame)
    J = ad.jacobian(chart_of(side.parent[parent_slot], frame), target)
    local = frame.at({inclusion.parameter_domain: ref})
    normal = np.real(ad.asarray(local.value(conormals[slot])))
    vector = np.linalg.pinv(J).T @ normal
    return vector / np.linalg.norm(vector)
