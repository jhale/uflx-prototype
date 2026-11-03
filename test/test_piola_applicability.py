"""One applicability decision is shared by gradients, pull-backs and contracts."""

import numpy as np
import pytest

from uflx_oracle import at, evaluate
from uflx_spec_interpreter import (
    RD,
    ScalarType,
    apply_gradients,
    coordinate,
    domain_map,
    expand_pull_backs,
    function,
    image,
    num,
    pull_back,
    tensor,
)
from uflx_spec_interpreter._draft import curl, divergence, push_forward, transformation_law
from uflx_spec_interpreter._src.domains import cell_domain
from uflx_spec_interpreter._src.nodes import Curl, Divergence, Gradient, PullBack, PushForward
from uflx_spec_interpreter._src.passes.gradients import allowed
from uflx_spec_interpreter._src.passes.piola import piola_law
from uflx_spec_interpreter._src.substitution import everywhere
from uflx_toy.cells import reference_simplex
from uflx_toy.spaces import space
from uflx_toy.topology import static_topology


def mapped(k, n, variance, weight):
    K = reference_simplex(k)
    T = static_topology((K,), ((tuple(range(k + 1)),),), ())
    x = function(
        space("piola-coordinates", value_shape=(n,), domain=T, scalar_type=ScalarType.real), "x"
    )
    D = image(x, RD(n))
    f = function(space("piola-field", value_shape=(k,), domain=T), "f")
    return D, push_forward(f, D, transformation_law((variance,), weight))


@pytest.mark.parametrize("k,n", [(2, 2), (3, 3), (2, 3)])
def test_divergence_identity_and_noncell_reads_agree_across_passes(k, n):
    D, f = mapped(k, n, "contravariant", 1)
    e = divergence(f, D)
    assert piola_law(e) is transformation_law((), 1)
    assert apply_gradients(e) is e
    assert allowed(e)
    # Coordinate-only maps into a mesh cannot select a cell.
    with pytest.raises(TypeError):
        domain_map(RD(0), tensor(tuple(num(0) for _ in range(n))), D)
    out = expand_pull_backs(pull_back(e, cell_domain(D, 0)))
    assert everywhere(out, lambda node: not isinstance(node, Divergence | PushForward))
    # The identity must not differentiate the geometric Jacobian.
    assert everywhere(out, lambda node: not isinstance(node, Gradient) or node.expression is not f)


@pytest.mark.parametrize("k,n", [(2, 2), (3, 3), (2, 3)])
def test_curl_eligibility_observes_the_dimension_restriction(k, n):
    D, f = mapped(k, n, "covariant", 0)
    e = curl(f, D)
    if k == n:
        expected = transformation_law(("contravariant",) if n == 3 else (), 1)
        assert piola_law(e) is expected
        assert apply_gradients(e) is e
        assert allowed(e)
    else:
        assert piola_law(e) is None
        assert not allowed(e)
        assert everywhere(apply_gradients(e), lambda node: not isinstance(node, Curl))


def test_wrong_mapping_laws_do_not_trigger_identities():
    D, f = mapped(2, 2, "covariant", 0)
    assert piola_law(divergence(f, D)) is None
    D, f = mapped(2, 2, "contravariant", 1)
    assert piola_law(curl(f, D)) is None


def test_ordinary_divergence_and_curl_expand_without_any_pullback():
    P = RD(2)
    x, y = coordinate(P)[0], coordinate(P)[1]
    cases = (
        (divergence(tensor((x * x, y * y)), P), 2 * 0.3 + 2 * 0.7),
        (curl(tensor((x * y, x * x)), P), 0.3),
    )
    for original, expected in cases:
        out = apply_gradients(original)
        assert everywhere(out, lambda node: not isinstance(node, Divergence | Curl | PullBack))
        actual = at(P, np.array([0.3, 0.7]), lambda: evaluate(out, {})(()))
        np.testing.assert_allclose(actual, expected)
