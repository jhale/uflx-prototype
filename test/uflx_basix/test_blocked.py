import basix
import pytest

from uflx_basix.cells import cell
from uflx_basix.spaces import space


def test_invalid_block_shapes():
    K = cell(basix.CellType.triangle)
    with pytest.raises(ValueError, match="positive"):
        space("bad", K, basix.ElementFamily.P, 1, shape=(0,))
    with pytest.raises(ValueError, match="scalar base"):
        space("bad", K, basix.ElementFamily.RT, 1, shape=(2,))
