"""Small meshes and named side selections shared by the numerical tests."""

import basix
import numpy as np

import uflx.geometry as geometry
from uflx.forms import function
from uflx.geometry import mesh
from uflx.nodes import EuclideanSpace, Mesh
from uflx_basix.cells import cell, topology
from uflx_basix.nodes import BasixCell
from uflx_basix.spaces import local_space

from ..uflx.spaces import space as field_space
from .cells import reference_simplex


def single_cell_mesh(cell_type):
    T = topology(cell_type)
    P1 = basix.create_element(basix.ElementFamily.P, cell_type, 1)
    x = function(local_space(T, P1, shape=(T.dimension,)), "x")
    return mesh(x, EuclideanSpace(T.dimension))


def vertex_mesh(T, vertices, *, label="x"):
    """Build a first-order coordinate coefficient and its cell-local bindings."""
    vertices = np.asarray(vertices, dtype=float)
    shape = (vertices.shape[1],)
    if isinstance(T.cell_types[0], BasixCell):
        elements = tuple(
            basix.create_element(basix.ElementFamily.P, K.cell_type, 1) for K in T.cell_types
        )
        X = local_space(T, elements, shape=shape)
        x = function(X, label)
        values = np.empty(X.num_dofs)
        for j, row in enumerate(T.cells):
            dofs = np.asarray(X.cell_dofs[j]).reshape(-1, shape[0])
            for vertex, (dof,) in zip(row, X.element(j).entity_dofs[0], strict=True):
                values[dofs[dof]] = vertices[vertex]
        bindings = {f"coefficients:{label}": values}
    else:
        # ReferenceSimplex fixtures supply an affine field on each reference cell.
        x = function(field_space("geometry", shape, T, scalar_type="real"), label)

        def chart(row):
            origin = vertices[row[0]]
            edges = vertices[list(row[1:])] - origin
            return lambda point: origin + point @ edges

        bindings = {x: {j: chart(row) for j, row in enumerate(T.cells)}}
    return mesh(x, EuclideanSpace(shape[0])), bindings


def push_forward(f, domain, law=None):
    """Use the toy function space's law when an example omits it."""
    return geometry.push_forward(f, domain, f.space.law if law is None else law)


# Named selections belong to these legacy test fixtures, not to topology nodes.
# The public constructor for an explicit relation is uflx.geometry.side_map.
_fixture_sides = {}


def side_map(domain, source, selection=None):
    T = domain.topology if isinstance(domain, Mesh) else domain
    S = source.topology if isinstance(source, Mesh) else source
    named = _fixture_sides.get(source, _fixture_sides[S])
    selected = selection if selection is not None else next(iter(named))
    relation = named[selected]
    if relation.parent != T:
        raise ValueError("fixture side has a different parent")
    return geometry.side_map(source, *((domain[j], A) for j, A in relation.entries))


def side_domain(domain, topology, selection=None):
    relation = side_map(domain.topology, topology, selection)
    result = mesh(geometry.pull_back(domain.coordinate_function, relation), domain.ambient_domain)
    _fixture_sides[result] = dict(_fixture_sides[topology])
    return result


def side_data(source, slot, name):
    from ..uflx.sides import outward_conormals

    relation = _fixture_sides[source][name]
    return relation.entries[slot][1], outward_conormals(relation)[slot]


def static_topology(cell_types, cells, sides=()):
    from ..uflx.library import static_topology as construct
    from ..uflx.sides import static_side_map

    source = construct(cell_types, cells)
    if sides:
        _fixture_sides[source] = {
            name: static_side_map(
                parent,
                source,
                *(next(j for n, T, j in entries if n == name and T == parent) for entries in sides),
            )
            for name, parent, _ in sides[0]
        }
    return source


def tabulated_topology(label, cell_types, sides, positions):
    from ..uflx.library import tabulated_topology as construct
    from ..uflx.sides import tabulated_side_map

    source = construct(label, cell_types)
    _fixture_sides[source] = {
        name: tabulated_side_map(
            parent,
            source,
            *(
                next(
                    (j, row[k]) for k, (n, T, j) in enumerate(entries) if n == name and T == parent
                )
                for entries, row in zip(sides, positions, strict=True)
            ),
        )
        for name, parent, _ in sides[0]
    }
    return source


def grouped_topology(cell_types, cells, sides):
    """Build fixed local slots from vertex groups used by whole-mesh test fixtures."""
    types, rows, declarations = [], [], []
    for group, (K, group_cells) in enumerate(zip(cell_types, cells, strict=True)):
        for vertices in group_cells:
            types.append(K)
            rows.append(tuple(vertices))
            chosen = []
            for name, parent, _ in sides[group] if sides else ():
                candidates = [
                    j for j, cell in enumerate(parent.cells) if set(vertices) <= set(cell)
                ]
                rank = sum(U is parent for _, U, _ in chosen)
                chosen.append((name, parent, candidates[min(rank, len(candidates) - 1)]))
            declarations.append(tuple(chosen))
    return static_topology(types, rows, tuple(declarations) if sides else ())


def square(K):
    # The square of two triangles on the reference cell K, with its diagonal and boundary.
    I = (
        cell(basix.CellType.interval)
        if K == cell(basix.CellType.triangle)
        else reference_simplex(1)
    )
    T = grouped_topology((K,), (((0, 1, 2), (0, 2, 3)),), ())
    D, bindings = vertex_mesh(T, [(0, 0), (1, 0), (1, 1), (0, 1)])
    diagonal = grouped_topology((I,), (((0, 2),),), ((("+", T, 0), ("-", T, 0)),))
    boundary = grouped_topology((I,), (((0, 1), (1, 2), (2, 3), (3, 0)),), ((("inside", T, 0),),))
    return (
        D,
        side_domain(D, diagonal),
        side_domain(D, boundary),
        bindings,
    )


def quadrilaterals():
    # The unit square as two quadrilaterals split at x = 1/2, with bilinear cell maps.
    Q, I = cell(basix.CellType.quadrilateral), cell(basix.CellType.interval)
    T = grouped_topology((Q,), (((0, 1, 3, 4), (1, 2, 4, 5)),), ())
    D, bindings = vertex_mesh(T, [(0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1), (1, 1)])
    interior = grouped_topology((I,), (((1, 4),),), ((("+", T, 0), ("-", T, 0)),))
    boundary = grouped_topology(
        (I,), (((0, 1), (1, 2), (2, 5), (5, 4), (4, 3), (3, 0)),), ((("inside", T, 0),),)
    )
    return (
        D,
        side_domain(D, interior),
        side_domain(D, boundary),
        bindings,
    )


def mixed():
    # The unit square as two triangles left of x = 1/2 and one quadrilateral right of it.
    Tri, Q, I = (
        cell(t)
        for t in (basix.CellType.triangle, basix.CellType.quadrilateral, basix.CellType.interval)
    )
    T = grouped_topology((Tri, Q), (((0, 1, 4), (0, 4, 3)), ((1, 2, 4, 5),)), ())
    D, bindings = vertex_mesh(T, [(0, 0), (0.5, 0), (1, 0), (0, 1), (0.5, 1), (1, 1)])
    # A group of facets between two triangles, and one between a triangle and a quadrilateral.
    interior = grouped_topology(
        (I, I),
        (((0, 4),), ((1, 4),)),
        ((("+", T, 0), ("-", T, 0)), (("+", T, 0), ("-", T, 1))),
    )
    boundary = grouped_topology(
        (I, I),
        (((0, 1), (4, 3), (3, 0)), ((1, 2), (2, 5), (5, 4))),
        ((("inside", T, 0),), (("inside", T, 1),)),
    )
    return (
        D,
        side_domain(D, interior),
        side_domain(D, boundary),
        bindings,
    )
