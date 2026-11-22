"""Concrete checks of mesh support versus explicit integration geometry."""

import basix
import numpy as np

from uflx.calculus import gradient, integral
from uflx.forms import function
from uflx.geometry import mesh
from uflx.nodes import EuclideanSpace
from uflx.tensors import tensor
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_compiler import compile_form
from uflx_compiler import evaluate as compiled_value

from ..uflx_basix.meshes import grouped_topology, push_forward, vertex_mesh


def test_coincident_cells_keep_independent_finite_element_fields():
    K = basix_cell(basix.CellType.interval)
    T = grouped_topology((K,), (((0, 1), (2, 3)),), ())
    D = vertex_mesh(T, [(0,), (1,), (0,), (1,)])
    V = basix_space("independent", T, basix.ElementFamily.P, 1)
    u = push_forward(function(V, "u"), D)
    coefficients = np.empty(V.num_dofs)
    for row, value in zip(V.cell_dofs, (2, 7), strict=True):
        coefficients[list(row)] = value
    result = compiled_value(compile_form(integral(u, D), 2), {"coefficients:u": coefficients})
    assert np.isclose(result, 9)
    assert D.topology is T


def test_diffusion_on_independent_analytic_curve_patches():
    # Two patch charts x_a(r) = (r, a*r**2), with a=1 and a=2.
    # No shared vertices or sides, and no new coordinate-expression class.
    K = basix_cell(basix.CellType.interval)
    T = grouped_topology((K,), (((0, 1), (2, 3)),), ())
    R = basix_space("patch-coordinate", T, basix.ElementFamily.P, 1)
    A = basix_space("patch-curvature", T, basix.ElementFamily.P, 0, discontinuous=True)
    r, a = function(R, "r"), function(A, "a")
    D = mesh(tensor((r, a * r * r)), EuclideanSpace(2))
    u = push_forward(r, D)
    g = gradient(u, D)
    energy = integral(g[0] * g[0] + g[1] * g[1], D)
    r_values = np.empty(R.num_dofs)
    a_values = np.empty(A.num_dofs)
    for row in R.cell_dofs:
        r_values[list(row)] = (0, 1)
    for row, curvature in zip(A.cell_dofs, (1, 2), strict=True):
        a_values[list(row)] = curvature
    env = {"coefficients:r": r_values, "coefficients:a": a_values}
    value = compiled_value(compile_form(energy, 40), env)
    # |grad_surface r|^2 ds = dr/sqrt(1 + 4*a*a*r*r).
    expected = sum(np.arcsinh(2 * curvature) / (2 * curvature) for curvature in (1, 2))
    assert np.isclose(value, expected, rtol=1e-10, atol=1e-12)
