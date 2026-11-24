# Copyright (C) 2026 Jack S. Hale
#
# This file is part of uflx-core-experimental (https://fenics.github.io/uflx/language/)
#
# SPDX-License-Identifier:    MIT
"""Regressions for rewrite completion, extension boundaries and value substitution."""

from __future__ import annotations

import pytest

from uflx._src.nodes import AbstractExpression, Base
from uflx._src.nodes.structure import expression_operands, operands
from uflx.forms import function
from uflx.nodes import Complex, Integer, ScalarType
from uflx.passes import lower

from .spaces import space


class ExternalOperation(AbstractExpression):
    identifier = "library.ExternalOperation"
    value_fields = ("operand",)
    operand: AbstractExpression
    value_shape = ()
    free_indices = ()
    argument_spaces = ()
    domain = None
    scalar_type = ScalarType.real


class LibraryTerminal(AbstractExpression):
    # Even a core-like identifier has no bearing on terminal status.
    identifier = "uflx.library.Terminal"
    is_terminal = True
    label: str
    value_shape = ()
    free_indices = ()
    argument_spaces = ()
    domain = None
    scalar_type = ScalarType.real


def test_unknown_library_operation_is_not_a_terminal():
    with pytest.raises(TypeError, match="ExternalOperation"):
        lower(ExternalOperation(Integer(2)))
    terminal = LibraryTerminal("data")
    assert lower(terminal) is terminal


def test_construction_dependencies_and_value_operands_are_distinct():
    V = space("dependencies")
    source = function(V, "source")
    f = function(V, ("derived", source))
    assert operands(f) == [V, source]
    assert expression_operands(f) == []
    c = Complex(Integer(1), Integer(2))
    assert operands(c) == [Integer(1), Integer(2)]
    assert expression_operands(c) == []


def test_validation_does_not_dispatch_on_class_name():
    class Index(Base):
        identifier = "library.Index"
        value: int

    assert Index(3).value == 3
