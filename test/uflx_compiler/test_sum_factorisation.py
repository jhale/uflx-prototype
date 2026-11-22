import basix
import jax
import numpy as np
import pytest

from uflx._src.nodes import GatheredIndex
from uflx.calculus import gradient, integral
from uflx.forms import function
from uflx_basix.assembly import conforming_dofmap
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space
from uflx_compiler import compile_form, evaluate, fold, tabulation
from uflx_oracle.interpreter import Interpreter

from ..testutil import contains
from ..uflx_basix.meshes import grouped_topology, push_forward, quadrilaterals, vertex_mesh
from .test_compiler import functionals

jax.config.update("jax_enable_x64", True)
P = basix.ElementFamily.P


def factored(compiled, dimension):
    # The coefficients are gathered with a dof index for each factor of the element.
    return contains(
        compiled, lambda n: isinstance(n, GatheredIndex) and len(n.indices) == dimension
    )


@pytest.mark.parametrize(
    ("degree", "discontinuous"), [(1, False), (2, False), (1, True), (3, True)]
)
def test_quadrilateral_kernels_factor_and_agree_with_the_oracle(degree, discontinuous):
    D, _, _, geometry = quadrilaterals()
    V = basix_space("V", D.topology, P, degree, discontinuous)
    uT = function(V, "u")
    u = push_forward(uT, D)
    mapping = conforming_dofmap(V)
    U = mapping.gather(np.random.default_rng(degree).normal(size=mapping.num_dofs))
    for e in functionals(D, u)[1:]:
        compiled = compile_form(e, 2 * degree + 2)
        assert factored(compiled, 2), e
        value = evaluate(compiled, geometry | {"coefficients:u": U}, np, fold(compiled))
        assert np.isclose(value, Interpreter(geometry | {uT: U}).evaluate(e), rtol=1e-8), e


def test_a_hexahedron_kernel_factors_and_agrees_with_the_oracle():
    H = basix_cell(basix.CellType.hexahedron)
    T = grouped_topology((H,), ((tuple(range(8)),),), ())
    # The unit cube shifted by (1, 2, 3), with vertex v at the bits of v, as Basix numbers them.
    D, geometry = vertex_mesh(
        T, [(1 + (v & 1), 2 + (v >> 1 & 1), 3 + (v >> 2 & 1)) for v in range(8)]
    )
    V = basix_space("V", D.topology, P, 2, True)
    uT = function(V, "u")
    u = push_forward(uT, D)
    g = gradient(u, D)
    e = integral(u * u + g[0] * g[1] + g[2] * g[2], D)
    U = np.random.default_rng(0).normal(size=V.num_dofs)
    compiled = compile_form(e, 6)
    assert factored(compiled, 3)
    value = evaluate(compiled, geometry | {"coefficients:u": U}, np, fold(compiled))
    assert np.isclose(value, Interpreter(geometry | {uT: U}).evaluate(e), rtol=1e-8)


def flops(e, U, geometry):
    compiled = compile_form(e, 14)
    folded = fold(compiled)
    kernel = jax.jit(
        lambda U: evaluate(compiled, geometry | {"coefficients:u": U}, jax.numpy, folded)
    )
    cost = kernel.lower(U).compile().cost_analysis()
    return (cost[0] if isinstance(cost, list) else cost)["flops"]


def test_factored_tables_take_fewer_operations(monkeypatch):
    D, _, _, geometry = quadrilaterals()
    V = basix_space("V", D.topology, P, 6, True)
    u = push_forward(function(V, "u"), D)
    g = gradient(u, D)
    e = integral(u * u + g[0] * g[0] + g[1] * g[1], D)
    U = np.random.default_rng(0).normal(size=V.num_dofs)
    fewer = flops(e, U, geometry)
    monkeypatch.setattr(tabulation, "separable", lambda points: False)
    assert fewer < flops(e, U, geometry) / 3
