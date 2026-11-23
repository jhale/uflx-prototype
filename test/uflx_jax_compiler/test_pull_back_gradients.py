import numpy as np

from uflx._src.nodes import Inverse
from uflx._src.passes.contracts import in_core_normal_form
from uflx._src.passes.lowering import lower_spatial
from uflx.arithmetic import num
from uflx.calculus import coordinate, gradient, integral
from uflx.geometry import domain_map, pull_back
from uflx.nodes import EuclideanSpace
from uflx.passes import expand_integrals
from uflx.tensors import component_tensor, index
from uflx_jax_compiler.interpreter import Interpreter

from ..uflx_basix.test_pull_back_gradients import GEOMETRY, D, K, nodes

X, k = coordinate(D), index("k")


def core(e):
    return lower_spatial(expand_integrals(e))


def test_a_gradient_pulled_back_by_a_map_that_is_not_a_cell_needs_no_inverse():
    # A coordinate-domain map uses the chain rule without inverting its Jacobian.
    P = EuclideanSpace(2)
    x = coordinate(P)
    phi = domain_map(P, component_tensor(x[k] + num(0.1) * x[0] * x[1], (k,)), P)
    f = pull_back(gradient(x[0] * x[0] * x[1], P), phi)[0]
    out = core(f)
    assert in_core_normal_form(out) and not any(isinstance(n, Inverse) for n in nodes(out))

    oracle = Interpreter()
    assert np.isclose(
        oracle.evaluate(out, points={P: np.array([0.2, 0.3])}),
        oracle.evaluate(f, points={P: np.array([0.2, 0.3])}),
    )


def test_a_gradient_of_the_coordinates_of_a_mesh_needs_no_inverse():
    # Gradients on the mesh come before the pull-backs by its cells, so K never meets J.
    f = integral(gradient(X[0] * X[0] * X[1], D)[0], D)
    out = core(f)
    assert in_core_normal_form(out) and not any(isinstance(n, Inverse) for n in nodes(out))
    oracle = Interpreter(GEOMETRY)
    assert np.isclose(oracle.evaluate(out), oracle.evaluate(f))


def test_cell_pullback_of_constant_gradient_introduces_no_geometry_inverse():
    from uflx._src.nodes import Zero
    from uflx.geometry import cell_map, push_forward, transformation_law
    from uflx.passes import expand_pull_backs
    from uflx.tensors import tensor

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
        oracle = Interpreter(GEOMETRY)
        np.testing.assert_allclose(
            oracle.evaluate(result),
            oracle.evaluate(original, points={K: np.array([0.2, 0.3])}),
            atol=1e-12,
        )
