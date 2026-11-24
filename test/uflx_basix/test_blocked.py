# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

import basix
import pytest

from uflx_basix.cells import cell
from uflx_basix.spaces import local_space


def test_invalid_block_shapes():
    K = cell(basix.CellType.triangle)
    with pytest.raises(ValueError, match="positive"):
        local_space(K, basix.create_element(basix.ElementFamily.P, K.cell_type, 1), shape=(0,))
    with pytest.raises(ValueError, match="scalar base"):
        local_space(K, basix.create_element(basix.ElementFamily.RT, K.cell_type, 1), shape=(2,))
