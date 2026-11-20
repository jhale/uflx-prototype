import basix
import numpy as np
from test_basix import square

from uflx_oracle import evaluate as oracle
from uflx_spec_interpreter import (
    EuclideanSpace,
    component_tensor,
    coordinate,
    domain_map,
    expand_integrals,
    gradient,
    index,
    integral,
    num,
    pull_back,
)
from uflx_spec_interpreter._src.algorithms.traversal import everywhere
from uflx_spec_interpreter._src.nodes import Inverse
from uflx_spec_interpreter._src.passes.contracts import in_core_normal_form
from uflx_spec_interpreter._src.passes.lowering import lower_spatial
from uflx_toy.basix_cells import basix_cell

K = basix_cell(basix.CellType.triangle)
D = square(K)[0]
X, k = coordinate(D), index("k")


def core(e):
    return lower_spatial(expand_integrals(e))


def nodes(e):
    found = []
    everywhere(e, lambda n: found.append(n) or True)
    return found


def test_a_gradient_pulled_back_by_a_map_that_is_not_a_cell_needs_no_inverse():
    # A coordinate-domain map uses the chain rule without inverting its Jacobian.
    P = EuclideanSpace(2)
    x = coordinate(P)
    phi = domain_map(P, component_tensor(x[k] + num(0.1) * x[0] * x[1], (k,)), P)
    f = pull_back(gradient(x[0] * x[0] * x[1], P), phi)[0]
    out = core(f)
    assert in_core_normal_form(out) and not any(isinstance(n, Inverse) for n in nodes(out))
    from uflx_oracle import at

    assert np.isclose(
        at(P, np.array([0.2, 0.3]), lambda: oracle(out, {})(())),
        at(P, np.array([0.2, 0.3]), lambda: oracle(f, {})(())),
    )


def test_a_gradient_of_the_coordinates_of_a_mesh_needs_no_inverse():
    # Gradients on the mesh come before the pull-backs by its cells, so K never meets J.
    f = integral(gradient(X[0] * X[0] * X[1], D)[0], D)
    out = core(f)
    assert in_core_normal_form(out) and not any(isinstance(n, Inverse) for n in nodes(out))
    assert np.isclose(oracle(out, {})(()), oracle(f, {})(()))


def test_cell_pullback_of_constant_gradient_introduces_no_geometry_inverse():
    from uflx_oracle import at
    from uflx_spec_interpreter import (
        cell_map,
        expand_pull_backs,
        push_forward,
        tensor,
        transformation_law,
    )
    from uflx_spec_interpreter._src.nodes import Zero

    for constant in (num(0), num(3), num(2j), tensor((num(1), num(2)))):
        mapped = push_forward(constant, D, transformation_law((), 0))
        original = pull_back(gradient(mapped, D), cell_map(D, 0))
        result = expand_pull_backs(original)
        assert isinstance(result, Zero)
        assert result.value_shape == original.value_shape
        assert result.free_indices == original.free_indices
        assert result.scalar_type == original.scalar_type
        assert result.argument_spaces == original.argument_spaces
        assert not any(isinstance(n, Inverse) for n in nodes(result))
        np.testing.assert_allclose(
            oracle(result, {})(()),
            at(K, np.array([0.2, 0.3]), lambda: oracle(original, {})(())),
            atol=1e-12,
        )


def test_constant_gradient_preserves_embedding_shape():
    from uflx_spec_interpreter import cell_map, expand_pull_backs, mesh, tensor
    from uflx_spec_interpreter._src.nodes import Zero

    embedded = mesh(
        tensor((D.coordinate_function[0], D.coordinate_function[1], num(0))), EuclideanSpace(3)
    )
    original = pull_back(gradient(num(2j), embedded), cell_map(embedded, 0))
    result = expand_pull_backs(original)
    assert isinstance(result, Zero)
    assert result.value_shape == (3,)
    assert result.free_indices == original.free_indices
    assert result.scalar_type == "complex"


def test_reference_constant_with_piola_mapping_is_not_a_physical_constant():
    from uflx_spec_interpreter import (
        cell_map,
        expand_pull_backs,
        push_forward,
        tensor,
        transformation_law,
    )
    from uflx_spec_interpreter._src.nodes import Zero

    mapped = push_forward(tensor((num(1), num(2))), D, transformation_law(("covariant",), 0))
    result = expand_pull_backs(pull_back(gradient(mapped, D), cell_map(D, 0)))
    assert not isinstance(result, Zero)
    assert any(isinstance(n, Inverse) for n in nodes(result))
