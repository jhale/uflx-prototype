from __future__ import annotations

from uflx._src.meshes import cell_map
from uflx.geometry import density, jacobian
from uflx.tensors import inverse

from ..testutil import contains
from .surface import M


def test_density_and_inverse_jacobian_share_the_jacobian():
    m = cell_map(M, 0)
    J = jacobian(m)
    assert contains(density(m), lambda n: n is J)
    assert contains(inverse(jacobian(m)), lambda n: n is J)
