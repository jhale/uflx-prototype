# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""The map contracts use fixed local slots, independently of physical coordinates."""

import numpy as np

from uflx.forms import function
from uflx.geometry import cell_map, pull_back, side_map
from uflx.passes import expand_pull_backs
from uflx_jax_compiler import trace

from ..uflx.spaces import space
from ..uflx_basix.meshes import side_data
from ..uflx_basix.test_map_slots import configuration


def test_side_map_resolves_both_parent_slots_without_indices():
    K, F, T, S = configuration()
    f = function(space("field", domain=T), "f")
    for side, slot, expected in (("+", 0, (0.6, 0.4)), ("-", 1, (0.0, 0.4))):
        A, _ = side_data(S, 0, side)
        sm = side_map(S, (T[slot], A))
        assert sm.construction_arguments == (T, S, ((slot, A),))
        assert sm.free_indices == ()
        assert A.parameter_domain is F and A.ambient_domain is K and A.free_indices == ()
        mapped = pull_back(f, sm)
        assert mapped.domain is S
        out = expand_pull_backs(pull_back(mapped, cell_map(S, 0)))
        assert out is pull_back(pull_back(f, cell_map(T, slot)), A)
        value = trace(A.parameterisation, F)(np.array([0.4]))
        np.testing.assert_allclose(value, expected)
