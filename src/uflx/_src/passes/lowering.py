# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""The standard core lowering recipe and its checked output boundary."""

from __future__ import annotations

from ..nodes import (
    AbstractExpression,
    Apply,
    Curl,
    DirectionalDerivative,
    Divergence,
    Division,
    Gradient,
    Integral,
    Product,
    PullBack,
    Sum,
)
from .applications import expand_applications
from .arithmetic import expand_arithmetic
from .contracts import core_normal_form
from .derivatives import apply_derivatives
from .integrals import expand_integrals
from .spatial import lower_spatial as spatial_fixed_point


def lower(f: AbstractExpression) -> AbstractExpression:
    """Lower a core expression and verify its core normal form.

    Geometry exposed during lowering must satisfy the geometry precondition of
    the specification. Extension operations require their own recipe before
    entering this operation. Retained library terminals are not evaluated here.

    Raises:
        TypeError: A rewrite is unsupported.
        ValueError: A stable core output violates the contract. Both report offending nodes
            in the exception's second argument.
    """
    out = expand_applications(f)
    out = apply_derivatives(out)
    out = expand_integrals(out)
    out = lower_spatial(out)
    out = expand_arithmetic(out)
    violations = core_normal_form(out)
    if violations:
        reasons = "; ".join(dict.fromkeys(reason for _, reason in violations))
        reducible = (
            Apply,
            Curl,
            DirectionalDerivative,
            Divergence,
            Division,
            Gradient,
            Integral,
            Product,
            PullBack,
            Sum,
        )
        if any(not isinstance(node, reducible) for node, _ in violations):
            raise TypeError(f"unsupported core lowering: {reasons}", violations)
        raise ValueError(f"stable output violates core normal form: {reasons}", violations)
    return out


def lower_spatial(f: AbstractExpression) -> AbstractExpression:
    """Resolve spatial operations and any derivatives they expose."""
    return spatial_fixed_point(f, apply_derivatives)
