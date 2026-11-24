# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Scalar categories in expression and function-space signatures."""

from __future__ import annotations

from typing import Final, Literal, TypeAlias

NumericalScalarType: TypeAlias = Literal["real", "complex"]
ExpressionScalarType: TypeAlias = Literal["boolean"] | NumericalScalarType


class ScalarType:
    boolean: Final = "boolean"
    real: Final = "real"
    complex: Final = "complex"


def numerical_type(scalar_type: ExpressionScalarType) -> NumericalScalarType:
    """Check the value-dependent part of a numerical operand's static type."""
    if scalar_type == ScalarType.real or scalar_type == ScalarType.complex:
        return scalar_type
    raise ValueError("numerical operand required")
