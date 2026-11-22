"""Local element coefficients and library continuity are independent choices."""

import basix
import numpy as np
import pytest

from uflx_basix.assembly import conforming_dofmap
from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space

from .meshes import square


@pytest.mark.parametrize("shape", [(2,), (2, 2)])
def test_blocked_fields_keep_cell_and_component_order(shape):
    D = square(basix_cell(basix.CellType.triangle))[0]
    V = basix_space("V", D.topology, basix.ElementFamily.P, 1, shape=shape)
    mapping = conforming_dofmap(V)
    values = np.arange(mapping.num_dofs)
    packed = mapping.gather(values).reshape(2, 3, -1)
    np.testing.assert_array_equal(packed[0, 0], packed[1, 0])
    np.testing.assert_array_equal(packed[0, 2], packed[1, 1])
    assert V.num_dofs == 6 * np.prod(shape)
