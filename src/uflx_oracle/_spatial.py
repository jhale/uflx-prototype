"""Numerical meanings of spatial derivatives, field mappings and conormals."""

import jax
import jax.numpy as jnp

from uflx.nodes import AbstractTopology, CellMap, Mesh, SideMap

from ._geometry import (
    chart_of,
    reference_point,
    side_reference,
    tangent_projector,
    transformed,
)
from ._tensors import inverse, size


def gradient(operand, domain, frame, arguments):
    point = jnp.asarray(frame.site.points[domain])
    if point.size == 0:
        shape = tuple(size(n, frame) for n in operand.value_shape)
        return jnp.empty((*shape, 0), dtype=complex if operand.scalar_type == "complex" else float)

    def field(p):
        value = frame.value(operand, arguments, at=frame.site.at({domain: p}))
        return jnp.asarray(value, dtype=jnp.result_type(value, 0.0))

    result = jax.jacfwd(field)(point)
    if isinstance(domain, Mesh) and domain.dimension < point.size:
        result = result @ tangent_projector(domain, frame)
    return result


def curl(operand, domain, frame, arguments):
    g = gradient(operand, domain, frame, arguments)

    def antisymmetric(a):
        if a.shape[-1] == 2:
            return a[1, 0] - a[0, 1]
        return jnp.array([a[2, 1] - a[1, 2], a[0, 2] - a[2, 0], a[1, 0] - a[0, 1]])

    return antisymmetric(g)


def pull_back(operand, mapping, frame, arguments):
    if isinstance(mapping, SideMap):
        _, slot, _, _, point = side_reference(mapping, frame)
        parent = mapping.parent
        if isinstance(parent, Mesh):
            point = chart_of(parent[slot], frame)(point)
        site = frame.site.at({parent: point}, cells={parent: slot})
    elif isinstance(mapping, CellMap):
        parent = mapping.parent
        point = (
            frame.site.points[mapping.reference_cell]
            if isinstance(parent, AbstractTopology)
            else jnp.real(frame.value(mapping.parameterisation))
        )
        site = frame.site.at({parent: point}, cells={parent: mapping.cell_slot})
    else:
        point = jnp.real(frame.value(mapping.parameterisation))
        site = frame.site.at({mapping.ambient_domain: point})
    return frame.value(operand, arguments, at=site)


def push_forward(operand, mesh, law, frame, arguments):
    slot = frame.site.slot(mesh)
    chart = chart_of(mesh[slot], frame)
    ref = reference_point(chart, frame.site.points[mesh], mesh.dimension)
    topology = mesh.topology
    site = frame.site.at({topology: ref}, cells={topology: slot})
    field = frame.value(operand, arguments, at=site)
    return transformed(field, law, jax.jacfwd(chart)(ref))


def conormal(side, conormals, frame):
    slot, parent_slot, inclusion, ref, target = side_reference(side, frame)
    J = jax.jacfwd(chart_of(side.parent[parent_slot], frame))(target)
    site = frame.site.at({inclusion.parameter_domain: ref})
    normal = jnp.real(frame.value(conormals[slot], at=site))
    vector = inverse(J).T @ normal
    return vector / jnp.linalg.norm(vector)
