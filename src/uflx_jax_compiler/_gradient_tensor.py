"""Factor homogeneous bilinear reference-gradient forms for P2 tetrahedral EmitC kernels."""

from functools import cache

import jax
import jax.numpy as jnp
import numpy as np

from uflx._src.algorithms.traversal import preorder
from uflx.algorithms import expression_operands, substitute_values
from uflx.nodes import (
    CellMap,
    ComponentTensor,
    Determinant,
    Division,
    Form,
    Function,
    Gradient,
    Indexed,
    IndexSum,
    MappedSpace,
    Product,
    PullBack,
    ReferenceValue,
    Sum,
    Tensor,
    Zero,
)
from uflx_extensions.runtime_arrays import runtime_array

from ._basis_tables import literal_values, sampled_quadrature
from ._finite_elements import basix, basix_space, vertex_p1_simplex
from .interpreter import trace


def _bilinear(expr, coefficient, test, trial):
    # Bit 0 denotes degree one in test, bit 1 degree one in trial.
    # Bit 2 tracks dependence on the coefficient at this point. A literal zero
    # (-1) is homogeneous in any slot. None denotes an unsupported expression.
    @cache
    def degree(node):
        if isinstance(node, Zero):
            return -1
        if node == test:
            return 1
        if node == trial:
            return 2
        if node == coefficient:
            return 4
        # The symbolic determinant derivative replaces one tensor row at a time.
        # Each determinant monomial takes exactly one entry from each row.
        if isinstance(node, Determinant) and isinstance(node.matrix, Tensor):
            rows = tuple(map(degree, node.matrix.entries))
            if None in rows:
                return None
            if -1 in rows:
                return -1
            result = 0
            for row in rows:
                if result & row & 3:
                    return None
                result |= row
            return result
        if (
            isinstance(node, Indexed)
            and isinstance(node.expression, Tensor)
            and isinstance(node.indices[0], int)
        ):
            return degree(node.expression.entries[node.indices[0]])
        degrees = tuple(map(degree, expression_operands(node)))
        if None in degrees:
            return None
        if not any(degrees):
            return 0
        if not node.is_pointwise:
            return None
        if isinstance(node, (Indexed, ComponentTensor, IndexSum)):
            return degrees[0]
        if isinstance(node, (Sum, Tensor)):
            degrees = tuple(d for d in degrees if d != -1)
            if not degrees:
                return -1
        if -1 in degrees:
            return -1 if isinstance(node, Product) else None
        coeff = 4 if any(d & 4 for d in degrees) else 0
        slots = tuple(d & 3 for d in degrees)
        if not any(slots):
            return coeff
        if isinstance(node, (Sum, Tensor)) and len(set(slots)) == 1:
            return coeff | slots[0]
        if isinstance(node, Division) and slots[1] == 0:
            return coeff | slots[0]
        if isinstance(node, Product) and not slots[0] & slots[1]:
            return coeff | slots[0] | slots[1]
        return None

    return degree(expr) == 7


def gradient_tensor_kernel(expression, inputs):
    """Return a node/component-blocked tangent, or defer to ordinary tabulation."""
    if not isinstance(expression, Form) or len(expression.arguments) != 2 or len(inputs) != 2:
        return None
    x, u = inputs
    fields = (u, *expression.arguments)
    if not all(isinstance(f, Function) and f.scalar_type == "real" for f in (x, *fields)):
        return None
    if len(set((x, *fields))) != 4:
        return None
    if not isinstance(u.space, MappedSpace) or any(f.space != u.space for f in fields):
        return None
    space = basix_space(u.space)
    if space is None or len(space.cell_dofs) != 1 or space.block_shape != (3,):
        return None

    element = space.element(0)
    if (
        element.family != basix.ElementFamily.P
        or element.degree != 2
        or element.cell_type != basix.CellType.tetrahedron
        or element.dtype != np.float64
        or x != u.space.mesh.coordinate_function
    ):
        return None
    geometry = basix_space(x.space)
    if (
        geometry is None
        or geometry.block_shape != (3,)
        or geometry.domain != space.domain
        or not vertex_p1_simplex(geometry.element(0))
    ):
        return None
    sampled = sampled_quadrature(expression.expression)
    if sampled is None:
        return None
    expr, cell, points, weights = sampled
    if (
        expr.argument_spaces
        or expr.free_indices
        or expr.value_shape
        or getattr(space.domain, "cell_types", ()) != (cell,)
    ):
        return None
    try:
        points = np.asarray(literal_values(points), dtype=element.dtype)
        weights = np.asarray(literal_values(weights), dtype=element.dtype)
    except ValueError:
        return None
    if len(weights) < 2:
        return None

    mapping = CellMap(space.domain, 0)
    gradients = [Gradient(PullBack(ReferenceValue(f, f.space), mapping), cell) for f in fields]
    slots = [runtime_array(("ufcx.reference_gradient", f.label), (3, 3)) for f in fields]
    integrand = substitute_values(expr, dict(zip(gradients, slots, strict=True)))

    nodes = set(preorder(integrand))
    if nodes.intersection(fields) or not set(slots) <= nodes or not _bilinear(integrand, *slots):
        return None
    evaluate = trace(integrand, x, cell, *slots)
    rows = element.tabulate(1, points)[1:4, :, :, 0].transpose(1, 2, 0)

    def kernel(coordinates, coefficients):
        unit = jnp.eye(9, dtype=coefficients.dtype).reshape(9, 3, 3)

        def point(q, basis):
            displacement_gradient = coefficients.reshape(element.dim, 3).T @ basis

            def component(test, trial):
                return evaluate(coordinates, q, displacement_gradient, test, trial)

            material = jax.vmap(jax.vmap(component, in_axes=(None, 0)), in_axes=(0, None))(
                unit, unit
            ).reshape(3, 3, 3, 3)
            # Contract reference directions first on the test side, then on the trial side.
            partial = sum(
                basis[:, r, None, None, None] * material[None, :, r, :, :] for r in range(3)
            )
            return sum(partial[:, :, None, :, s] * basis[None, None, :, s, None] for s in range(3))

        values = jax.vmap(point)(jnp.asarray(points), jnp.asarray(rows))
        return jnp.tensordot(weights, values, axes=1)

    kernel.inputs = inputs
    return kernel
