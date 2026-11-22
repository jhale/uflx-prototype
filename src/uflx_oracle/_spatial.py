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
    if np.size(ad.primal(point)) == 0:
        shape = tuple(size(n, frame) for n in operand.value_shape)
        return np.empty((*shape, 0), dtype=complex if operand.scalar_type == "complex" else float)

    def field(p):
        return ad.asarray(frame.at({domain: p}).value(operand, arguments))

    result = ad.jacobian(field, point)
    if isinstance(domain, Mesh) and domain.dimension < np.size(ad.primal(point)):
        result = ad.matmul(result, tangent_projector(domain, frame))
    return result


def curl(operand, domain, frame, arguments):
    g = gradient(operand, domain, frame, arguments)

    def antisymmetric(a):
        if a.shape[-1] == 2:
            return a[1, 0] - a[0, 1]
        return np.array([a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]])

    return ad.linear(antisymmetric, g)


def pull_back(operand, mapping, frame, arguments):
    if isinstance(mapping, SideMap):
        _, slot, _, _, point = side_reference(mapping, frame)
        parent = mapping.parent
        if isinstance(parent, Mesh):
            point = chart_of(parent[slot], frame)(point)
        local = frame.at({parent: point}, cells={parent: slot})
    elif isinstance(mapping, CellMap):
        parent = mapping.parent
        point = (
            frame.points[mapping.reference_cell]
            if isinstance(parent, AbstractTopology)
            else ad.linear(np.real, frame.value(mapping.parameterisation))
        )
        local = frame.at({parent: point}, cells={parent: mapping.cell_slot})
    else:
        point = ad.linear(np.real, frame.value(mapping.parameterisation))
        local = frame.at({mapping.ambient_domain: point})
    return local.value(operand, arguments)


def push_forward(operand, mesh, law, frame, arguments):
    slot = frame.cells.get(mesh, 0)
    chart = chart_of(mesh[slot], frame)
    ref = reference_point(chart, frame.points[mesh], mesh.dimension)
    topology = mesh.topology
    field = frame.at({topology: ref}, cells={topology: slot}).value(operand, arguments)
    return transformed(field, law, ad.jacobian(chart, ref))


def conormal(side, conormals, frame):
    slot, parent_slot, inclusion, ref, target = side_reference(side, frame)
    J = ad.jacobian(chart_of(side.parent[parent_slot], frame), target)
    local = frame.at({inclusion.parameter_domain: ref})
    normal = ad.linear(np.real, local.value(conormals[slot]))
    vector = ad.matmul(ad.linear(np.transpose, ad.pinv(J)), normal)
    return ad.divide(vector, ad.norm(vector))
