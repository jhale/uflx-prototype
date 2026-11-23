"""Concrete checks of mesh support versus explicit integration geometry."""

import numpy as np

from uflx._src.algorithms.substitution import substitute
from uflx._src.passes.lowering import lower_spatial
from uflx.arithmetic import num
from uflx.calculus import coordinate, integral
from uflx.forms import function
from uflx.geometry import density, domain_map, pull_back
from uflx.nodes import EuclideanSpace
from uflx.tensors import index, index_sum, tensor
from uflx_jax_compiler.interpreter import Interpreter

from ..testutil import table
from ..uflx.spaces import space
from ..uflx_basix.cells import reference_simplex
from ..uflx_basix.meshes import grouped_topology, vertex_mesh


def test_integrate_members_then_sum_only_the_selected_index():
    P = reference_simplex(1)
    i, configuration = index("segment"), index("configuration")
    lengths = table([[2, 4], [3, 6], [5, 10]])
    M = domain_map(P, tensor((lengths[i, configuration] * coordinate(P)[0],)), EuclideanSpace(1))
    members = integral(pull_back(num(1), M) * density(M), P)
    total = index_sum(members, (i,))
    assert dict(members.free_indices) == {i: 3, configuration: 2}
    assert dict(total.free_indices) == {configuration: 2}
    for c, expected in enumerate((10, 20)):
        assert np.isclose(
            Interpreter().evaluate(lower_spatial(total), indices={configuration: c}), expected
        )


def test_integration_patches_need_no_field_domain():
    P = reference_simplex(1)
    t = coordinate(P)[0]
    patches = (
        domain_map(P, tensor((num(2) * t,)), EuclideanSpace(1)),
        domain_map(P, tensor((num(2) + num(3) * t,)), EuclideanSpace(1)),
    )
    f = coordinate(EuclideanSpace(1))[0]
    expression = sum((integral(pull_back(f, M) * density(M), P) for M in patches), num(0))
    assert np.isclose(Interpreter().evaluate(lower_spatial(expression)), 12.5)


def test_mesh_integral_retains_integrand_indices_across_substitution():
    from uflx._src.passes.integrals import expand_integrals

    K = reference_simplex(1)
    T = grouped_topology((K,), (((0, 1), (2, 3)),), ())
    D, geometry = vertex_mesh(T, [(0,), (1,), (0,), (1,)])
    c = index("load-entry")
    r = function(space("load"), "r")
    load = table([2, 7])[c]
    expression = integral(r, D)
    inserted = substitute(expression, {r: load})
    first = expand_integrals(inserted)
    second = substitute(expand_integrals(expression), {r: load})
    assert inserted.free_indices == first.free_indices == second.free_indices == ((c, 2),)
    for n, expected in enumerate((4, 14)):
        for form in (inserted, first, second):
            assert np.isclose(
                Interpreter(geometry).evaluate(lower_spatial(form), indices={c: n}), expected
            )
    assert expand_integrals(inserted) is first
