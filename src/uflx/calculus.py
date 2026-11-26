# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Differentiation and integration of UFLx expressions."""

from ._src.calculus import (
    coordinate,
    curl,
    directional_derivative,
    divergence,
    gradient,
    integral,
    linearize,
)
from ._src.rules.integration import reference_integral

__all__ = [
    "coordinate",
    "curl",
    "directional_derivative",
    "divergence",
    "gradient",
    "integral",
    "linearize",
    "reference_integral",
]
