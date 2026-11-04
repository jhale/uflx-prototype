"""Diagnostic contracts are read-only checks, independent of lowering recipes."""

from uflx_spec_interpreter import (
    RD,
    ScalarType,
    apply_derivatives,
    directional_derivative,
    function,
    gradient,
    integral,
    num,
    tensor,
)
from uflx_spec_interpreter._src.nodes import DirectionalDerivative
from uflx_spec_interpreter._src.passes.contracts import (
    _core_normal_form,
    _derivatives_expanded,
    _in_core_normal_form,
    _reference_integration,
)
from uflx_toy.runtime import runtime_array
from uflx_toy.spaces import space


def test_reports_shared_nodes_once_and_does_not_lower():
    u, h = (function(space("contract"), name) for name in ("u", "h"))
    d = directional_derivative(u * u, u, u, h)
    e = tensor((d, d))
    assert _derivatives_expanded(e) == ((d, "directional derivative remains"),)
    assert isinstance(e.entries[0], DirectionalDerivative)
    assert not _derivatives_expanded(apply_derivatives(e))


def test_reports_multiple_failures_in_stable_order():
    u, h = (function(space("contract-order"), name) for name in ("u", "h"))
    first = directional_derivative(u, u, u, h)
    second = directional_derivative(u * u, u, u, h)
    e = tensor((first, second, first))
    assert tuple(n for n, _ in _derivatives_expanded(e)) == (first, second)
    assert _derivatives_expanded(e) == _derivatives_expanded(e)


def test_later_compiler_capabilities_are_not_checked():
    # An abstract coordinate-domain integral need not have a Basix quadrature rule.
    e = integral(num(1), RD(2))
    assert not _reference_integration(e)
    assert not _core_normal_form(e)
    assert not _core_normal_form(runtime_array("external-data", (2,)))


def test_nonexpanded_gradient_is_allowed_at_earlier_boundaries():
    P = RD(2)
    u = function(space("gradient-contract", domain=P), "u")
    e = gradient(u * u, P)
    assert not _derivatives_expanded(e)
    assert not _reference_integration(e)
    assert _core_normal_form(e) == ((e, "Gradient is outside core normal form"),)
    assert bool(_core_normal_form(e)) == (not _in_core_normal_form(e))
    assert not _core_normal_form(gradient(u, P))


def test_contract_stops_at_stored_geometry_and_reports_mesh_integrals():
    import basix
    from test_basix import square

    from uflx_spec_interpreter._src.meshes import image
    from uflx_toy.basix_cells import basix_cell

    D = square(basix_cell(basix.CellType.triangle))[0]
    q = function(
        space("coordinates", value_shape=(2,), domain=D.topology, scalar_type=ScalarType.real), "q"
    )
    coordinates = directional_derivative(q, q, D.coordinate_function, q)
    geometry = image(coordinates, D.ambient_domain)
    u = function(space("geometry-contract", domain=geometry), "u")
    assert not _derivatives_expanded(u)
    assert not _core_normal_form(u)
    assert _derivatives_expanded(coordinates)
    e = integral(u, geometry)
    assert not _derivatives_expanded(e)
    assert _reference_integration(e) == ((e, "mesh integral remains"),)
    assert _core_normal_form(e) == ((e, "Integral is outside core normal form"),)
