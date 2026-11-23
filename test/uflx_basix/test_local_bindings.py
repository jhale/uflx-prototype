"""Local element coefficients and library continuity are independent choices."""

import basix
import numpy as np
import pytest

from uflx_basix.cells import cell
from uflx_basix.spaces import space

from .meshes import square


@pytest.mark.parametrize("shape", [(2,), (2, 2)])
def test_blocked_fields_keep_cell_and_component_order(shape):
    D = square(cell(basix.CellType.triangle))[0]
    V = space(
        D.topology,
        basix.create_element(basix.ElementFamily.P, basix.CellType.triangle, 1),
        shape=shape,
    )
    per_cell = 3 * int(np.prod(shape))
    assert V.cell_dofs == (tuple(range(per_cell)), tuple(range(per_cell, 2 * per_cell)))
    assert V.num_dofs == 2 * per_cell
