# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT

from uflx.arithmetic import num
from uflx.tensors import component_tensor, index_sum, indexed


def test_scalar_empty_index_identities():
    scalar = num(2)
    assert indexed(scalar, ()) is scalar
    assert component_tensor(scalar, ()) is scalar
    assert index_sum(scalar, ()) is scalar
