"""Symbolic preparation shared by compiler targets before quadrature and tracing."""

from uflx._src.algorithms.substitution import occurs_in_metadata
from uflx._src.algorithms.traversal import preorder
from uflx.algorithms import argument_leaves, bottom_up, map_values, occurs_in_values
from uflx.geometry import jacobian
from uflx.nodes import (
    AbstractExpression,
    CellMap,
    Determinant,
    DirectionalDerivative,
    Mesh,
    PullBack,
)

from ._jacobian_products import cancel_jacobian_products


def lower(f: AbstractExpression) -> AbstractExpression:
    """Lower a compiler input with cell-map contractions and small determinant derivatives."""
    from uflx._src.passes.lowering import lower_spatial
    from uflx.passes import expand_applications, expand_integrals
    from uflx.passes import lower as core_lower

    expanded = expand_applications(f)
    meshes = set()
    for node in preorder(expanded):
        if isinstance(node.domain, Mesh):
            meshes.add(node.domain)
        if (
            isinstance(node, PullBack)
            and isinstance(node.map, CellMap)
            and isinstance(node.map.parent, Mesh)
        ):
            meshes.add(node.map.parent)
    jacobians = {jacobian(cell) for mesh in meshes for cell in mesh}
    jacobians = {
        J for J in jacobians if not J.free_indices and 0 < J.value_shape[1] <= J.value_shape[0]
    }

    def prepare(node: AbstractExpression) -> AbstractExpression:
        if not isinstance(node, DirectionalDerivative):
            return node
        variables = argument_leaves(node.variable)

        def prepare_body(body):
            # Shape AD already exposes coordinate-dependent geometry (D21-D23).
            # Cancel before differentiating the inverse produces extra terms.
            body = cancel_jacobian_products(body, jacobians)
            if jacobians and any(occurs_in_metadata(v, body) for v in variables):
                exposed = lower_spatial(expand_integrals(body))
                reduced = cancel_jacobian_products(exposed, jacobians)
                if reduced != exposed:
                    body = reduced
            return bottom_up(body, expand)

        def expand(det: AbstractExpression) -> AbstractExpression:
            if not isinstance(det, Determinant) or det.matrix.value_shape not in ((2, 2), (3, 3)):
                return det
            matrix = det.matrix
            if matrix in jacobians or not any(
                occurs_in_values(variable, matrix) for variable in variables
            ):
                return det

            if matrix.value_shape == (2, 2):
                return matrix[0, 0] * matrix[1, 1] - matrix[0, 1] * matrix[1, 0]

            # Expand along the first row. Cyclic column order includes the minor signs.
            def term(j: int, k: int, l: int) -> AbstractExpression:
                return matrix[0, j] * (matrix[1, k] * matrix[2, l] - matrix[1, l] * matrix[2, k])

            return term(0, 1, 2) + term(1, 2, 0) + term(2, 0, 1)

        return map_values(node, prepare_body, ("expression",))

    lowered = core_lower(bottom_up(expanded, prepare))
    return cancel_jacobian_products(lowered, jacobians)
