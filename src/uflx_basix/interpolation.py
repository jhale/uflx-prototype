"""Symbolic interpolation using library basis data and core mapping operations."""

from functools import reduce
from operator import add

from uflx.calculus import integral
from uflx.geometry import domain_map, euclidean_space, pull_back, push_forward, transformation_law
from uflx.nodes import AbstractExpression, CellMap, Function, Mesh, PullBack
from uflx.tensors import component_tensor, index, index_sum

from .nodes import BasixBasis, BasixSpace, table


def interpolate(
    f: AbstractExpression, V: BasixSpace, *, cell: CellMap | None = None, mesh: Mesh | None = None
) -> AbstractExpression:
    """Interpolate on one cell with the target element's value DOFs.

    A target bound with local_space(mesh[i], element) or V.on_cell(mesh[i]),
    or a source restricted to a physical cell, supplies the piece. Local input
    and output values use physical components in reference coordinates. A topology-wide
    target with a mesh field instead returns a mesh field supported on cell=,
    which can be inferred on a single-cell mesh.
    """
    if not isinstance(V, BasixSpace):
        raise ValueError("interpolation requires a Basix space")
    if f.argument_spaces:
        raise ValueError("function inputs must be empty")
    if f.scalar_type != V.scalar_type:
        raise ValueError("input and target scalar types must match")

    source_cell = None
    if isinstance(f, PullBack) and isinstance(f.map, CellMap) and isinstance(f.map.parent, Mesh):
        source_cell = f.map
    elif isinstance(f, Function) and isinstance(f.space, BasixSpace):
        source_cell = f.space.cell
    restricted = V.cell is not None or source_cell is not None
    for restriction in (V.cell, source_cell):
        if restriction is not None:
            if cell is not None and cell != restriction:
                raise ValueError("interpolation restrictions select different cells")
            cell = restriction

    if cell is None:
        mesh = f.domain if mesh is None else mesh
        if not isinstance(mesh, Mesh):
            raise ValueError("interpolation requires a physical mesh cell")
        if len(mesh.topology.cell_types) != 1:
            raise ValueError(
                "select a cell by restricting the target or field, or pass cell=mesh[i]"
            )
        cell = mesh[0]
    if not isinstance(cell, CellMap) or not isinstance(cell.parent, Mesh):
        raise ValueError("cell must be a physical mesh cell")
    if mesh is not None and mesh != cell.parent:
        raise ValueError("cell must belong to the supplied mesh")
    mesh = cell.parent
    K = cell.reference_cell
    if restricted:
        V = V.on_cell(cell)
    elif V.domain != mesh.topology:
        raise ValueError("the target space must be on the mesh topology")
    law = V.law
    unmapped = len(V.value_shape) - len(law.variances)
    physical_shape = V.value_shape[:unmapped] + (mesh.ambient_domain.dimension,) * len(
        law.variances
    )
    if f.value_shape != physical_shape:
        raise ValueError("input and physical target value shapes must match")

    if f.domain == mesh.ambient_domain:
        values = pull_back(f, domain_map(K, cell.parameterisation, mesh.ambient_domain))
    elif f.domain in (None, mesh):
        values = pull_back(f, cell)
    elif restricted and f.domain == K:
        values = f
    else:
        raise ValueError(
            "input domain must match the mesh, its ambient domain or the selected cell"
        )
    result = push_forward(_interpolate_cell(values, V, cell), mesh, law)
    return pull_back(result, cell) if restricted else result


def _interpolate_cell(f, V, cell):
    basis = BasixBasis(V, cell.cell_slot)
    E = basis.element
    if E.interpolation_nderivs:
        raise NotImplementedError("interpolation with derivative DOFs is not supported")

    # A DOF weight pairs with a field value, so its axes have the opposite variance.
    law = V.law
    dual_law = transformation_law(
        tuple("contravariant" if v == "covariant" else "covariant" for v in law.variances),
        -law.weight,
    )
    a = index(("interpolate", f, basis, "dof"))

    def axes(role, shape):
        return tuple(index(("interpolate", f, basis, role, k)) for k in range(len(shape)))

    blocks = axes("block", V.block_shape)
    physical = axes("physical", E.value_shape)
    reference = axes("reference", E.value_shape)

    # Basix columns run over points within each value component.
    weights = E.interpolation_matrix.reshape(E.dim, *E.value_shape, len(E.points))
    point_domain = euclidean_space(0)

    def sample(q, point):
        directions = push_forward(table(weights[..., q]), cell.parent, dual_law)
        physical_weights = pull_back(directions, cell)
        pairing = index_sum(physical_weights[(a, *physical)] * f[(*blocks, *physical)], physical)
        at_point = domain_map(point_domain, table(point), cell.reference_cell)
        return integral(pull_back(pairing, at_point), point_domain)

    dofs = reduce(add, (sample(q, point) for q, point in enumerate(E.points)))
    return component_tensor(index_sum(dofs * basis[(a, *reference)], (a,)), (*blocks, *reference))
