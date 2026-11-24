# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Tensor construction, indexing and contractions."""

from ._src.indices import gathered_index, index
from ._src.tensors import (
    broadcast_to,
    component,
    component_tensor,
    determinant,
    identity,
    index_sum,
    indexed,
    inverse,
    matrix_product,
    tensor,
    transpose,
)

__all__ = [
    "broadcast_to",
    "component",
    "component_tensor",
    "determinant",
    "gathered_index",
    "identity",
    "index",
    "index_sum",
    "indexed",
    "inverse",
    "matrix_product",
    "tensor",
    "transpose",
]
