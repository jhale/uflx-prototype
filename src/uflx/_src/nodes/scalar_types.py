# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Scalar categories in expression and function-space signatures."""

from __future__ import annotations

from typing import TYPE_CHECKING, Final

from ..typing import ExpressionScalarType, NumericalScalarType

if TYPE_CHECKING:
    from .expressions import AbstractExpression


class ScalarType:
    boolean: Final = "boolean"
    real: Final = "real"
    complex: Final = "complex"


def common_scalar_type(*operands: AbstractExpression) -> ExpressionScalarType:
    """Return the common scalar type of the operands."""
    types = {operand.scalar_type for operand in operands}
    if ScalarType.boolean in types:
        if len(types) != 1:
            raise ValueError("Boolean and numerical values cannot be mixed")
        return ScalarType.boolean
    return ScalarType.complex if ScalarType.complex in types else ScalarType.real


def numerical_type(scalar_type: ExpressionScalarType) -> NumericalScalarType:
    """Check the value-dependent part of a numerical operand's static type."""
    if scalar_type == ScalarType.real or scalar_type == ScalarType.complex:
        return scalar_type
    raise ValueError("numerical operand required")
