"""One applicability decision is shared by gradients, pull-backs and contracts."""

import pytest

from uflx._src.algorithms.traversal import everywhere
from uflx._src.meshes import cell_map
from uflx._src.nodes import Curl, Divergence, Gradient, PushForward
from uflx._src.passes.contracts import _allowed_in_core_normal_form
from uflx._src.rules.coordinates import piola_law
from uflx.calculus import curl, divergence
from uflx.forms import function
from uflx.geometry import mesh, pull_back, transformation_law
from uflx.nodes import EuclideanSpace, ScalarType
from uflx.passes import apply_gradients, expand_pull_backs
from uflx_basix.cells import reference_simplex
from uflx_basix.spaces import space

from .meshes import grouped_topology, push_forward


def mapped(k, n, variance, weight):
    K = reference_simplex(k)
    T = grouped_topology((K,), ((tuple(range(k + 1)),),), ())
    x = function(
        space("piola-coordinates", value_shape=(n,), domain=T, scalar_type=ScalarType.real), "x"
    )
    D = mesh(x, EuclideanSpace(n))
    f = function(space("piola-field", value_shape=(k,), domain=T), "f")
    return D, push_forward(f, D, transformation_law((variance,), weight))


@pytest.mark.parametrize("k,n", [(2, 2), (3, 3), (2, 3)])
def test_divergence_identity_and_noncell_reads_agree_across_passes(k, n):
    D, f = mapped(k, n, "contravariant", 1)
    e = divergence(f, D)
    assert piola_law(e) is transformation_law((), 1)
    assert apply_gradients(e) is e
    assert _allowed_in_core_normal_form(e)
    out = expand_pull_backs(pull_back(e, cell_map(D, 0)))
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
        assert _allowed_in_core_normal_form(e)
    else:
        assert piola_law(e) is None
        assert not _allowed_in_core_normal_form(e)
        assert everywhere(apply_gradients(e), lambda node: not isinstance(node, Curl))


def test_wrong_mapping_laws_do_not_trigger_identities():
    D, f = mapped(2, 2, "covariant", 0)
    assert piola_law(divergence(f, D)) is None
    D, f = mapped(2, 2, "contravariant", 1)
    assert piola_law(curl(f, D)) is None
