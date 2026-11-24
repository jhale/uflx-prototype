# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Diagnostic contracts are read-only checks, independent of lowering recipes."""

from uflx._src.passes.contracts import core_normal_form, derivatives_expanded, reference_integration
from uflx.arithmetic import num
from uflx.calculus import directional_derivative, integral
from uflx.forms import function
from uflx.nodes import EuclideanSpace, ScalarType

from ..uflx.library import runtime_array
from ..uflx.spaces import space


def test_later_compiler_capabilities_are_not_checked():
    # An abstract coordinate-domain integral need not have a Basix quadrature rule.
    e = integral(num(1), EuclideanSpace(2))
    assert not reference_integration(e)
    assert not core_normal_form(e)
    assert not core_normal_form(runtime_array("external-data", (2,)))


def test_contract_stops_at_stored_geometry_and_reports_mesh_integrals():
    import basix

    from uflx._src.meshes import create_mesh
    from uflx_basix.cells import cell

    from .meshes import square

    D = square(cell(basix.CellType.triangle))[0]
    q = function(
        space("coordinates", D.topology, value_shape=(2,), scalar_type=ScalarType.real), "q"
    )
    coordinates = directional_derivative(q, q, D.coordinate_function, q)
    geometry = create_mesh(coordinates, D.ambient_domain)
    u = function(space("geometry-contract", geometry), "u")
    assert not derivatives_expanded(u)
    assert not core_normal_form(u)
    assert derivatives_expanded(coordinates)
    e = integral(u, geometry)
    assert not derivatives_expanded(e)
    assert reference_integration(e) == ((e, "mesh integral remains"),)
    assert core_normal_form(e) == ((e, "Integral is outside core normal form"),)
