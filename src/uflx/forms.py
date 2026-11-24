# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Functions, function spaces, form arguments and application."""

from ._src.algorithms.substitution import rebind
from ._src.forms import apply, form, function, partial
from ._src.spaces import conjugate_space, dual_space

__all__ = [
    "apply",
    "conjugate_space",
    "dual_space",
    "form",
    "function",
    "partial",
    "rebind",
]
