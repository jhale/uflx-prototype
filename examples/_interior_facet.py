"""Interior-edge geometry for the two-triangle DG example."""

import basix
import jax.numpy as jnp
import numpy as np

from uflx.forms import function
from uflx.geometry import domain_map, side_domain, side_map
from uflx_basix.cells import cell, topology
from uflx_basix.spaces import local_space


def interior_facet(plus_cell, minus_cell):
    """Return the edge and its side maps, and the reference geometry for the facet kernel."""
    interval = basix.CellType.interval
    I, K = cell(interval), cell(basix.CellType.triangle)
    p1 = basix.create_element(basix.ElementFamily.P, interval, 1)
    p0 = basix.create_element(basix.ElementFamily.P, interval, 0, discontinuous=True)

    # Each chart joins two vertices of the reference triangle. The P2 cell map curves the edge.
    charts = local_space(I, p1, shape=(2,))
    plus_chart = function(charts, "plus chart")
    minus_chart = function(charts, "minus chart")
    into_plus = domain_map(I, plus_chart, K)
    into_minus = domain_map(I, minus_chart, K)
    edge = side_domain(plus_cell.parent, topology(interval), (plus_cell.cell_slot, into_plus))
    plus = side_map(edge, (plus_cell, into_plus))
    minus = side_map(edge, (minus_cell, into_minus))
    ref_normal = function(local_space(I, p0, shape=(2,)), "plus reference normal")
    return (edge, plus, minus), (plus_chart, minus_chart, ref_normal)


def facet_geometry_values(local_facets, permutations):
    """Look up the two charts and the plus reference normal in trace-input order."""
    triangle = basix.CellType.triangle
    edges = np.asarray(basix.cell.topology(triangle)[1])
    points = basix.cell.geometry(triangle)[edges]
    oriented_edges = jnp.asarray(np.stack((points, points[:, ::-1]), axis=1))
    normals = jnp.asarray(basix.cell.facet_outward_normals(triangle))
    plus_points, minus_points = oriented_edges[local_facets, permutations]
    return plus_points, minus_points, normals[local_facets[0]]
