import basix
import pytest

from uflx_basix.basix_cells import basix_cell
from uflx_basix.basix_spaces import basix_space


def test_invalid_block_shapes():
    cell = basix_cell(basix.CellType.triangle)
    with pytest.raises(ValueError, match="positive"):
        basix_space("bad", cell, basix.ElementFamily.P, 1, shape=(0,))
    with pytest.raises(ValueError, match="scalar base"):
        basix_space("bad", cell, basix.ElementFamily.RT, 1, shape=(2,))
