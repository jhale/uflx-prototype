from __future__ import annotations

from functools import cached_property
from typing import final

from .arithmetic import Conjugate
from .base import AbstractExpression
from .spaces import AbstractFunctionSpace


def entry_space(entry):
    # The space of v, the conjugate of the space for conjugate(v), and a tuple for a tuple entry.
    from ..spaces import conjugated

    if isinstance(entry, tuple):
        return tuple(map(entry_space, entry))
    return conjugated(entry.operand.space) if isinstance(entry, Conjugate) else entry.space


@final
class Function(AbstractExpression):
    identifier = "uflx.Function"
    space: AbstractFunctionSpace
    label: str | tuple
    opaque = ("space", "label")
    labels = ("label",)

    @cached_property
    def value_shape(self):
        return self.space.value_shape

    @cached_property
    def argument_spaces(self):
        return self.space.argument_spaces

    @cached_property
    def scalar_type(self):
        return self.space.scalar_type

    @cached_property
    def support(self):
        return self.space.domain

    @cached_property
    def free_indices(self):
        return ()


@final
class Form(AbstractExpression):
    identifier = "uflx.Form"
    expression: AbstractExpression
    arguments: tuple
    scope = ("expression", "arguments")

    def bound(self):
        # The functions of the entries, without the conjugate of an entry conjugate(v).
        members = [m for a in self.arguments for m in (a if isinstance(a, tuple) else (a,))]
        return [m.operand if isinstance(m, Conjugate) else m for m in members]

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def argument_spaces(self):
        # The arguments of the expression come first, then the entries.
        return self.expression.argument_spaces + tuple(map(entry_space, self.arguments))

    @cached_property
    def free_indices(self):
        return self.expression.free_indices

    @cached_property
    def scalar_type(self):
        return self.expression.scalar_type

    @cached_property
    def support(self):
        return self.expression.support


@final
class Action(AbstractExpression):
    identifier = "uflx.Action"
    form: AbstractExpression
    position: int
    argument: AbstractExpression | tuple[AbstractExpression, ...]
    opaque = ("position",)

    @cached_property
    def value_shape(self):
        return self.form.value_shape

    @cached_property
    def argument_spaces(self):
        spaces = self.form.argument_spaces
        return spaces[: self.position] + spaces[self.position + 1 :]

    @cached_property
    def free_indices(self):
        members = self.argument if isinstance(self.argument, tuple) else (self.argument,)
        return tuple(dict(p for x in (self.form, *members) for p in x.free_indices).items())

    @cached_property
    def scalar_type(self):
        return self.form.scalar_type

    @cached_property
    def support(self):
        return self.form.support


@final
class GateauxDerivative(AbstractExpression):
    identifier = "uflx.GateauxDerivative"
    expression: AbstractExpression
    variable: Function | tuple[Function, ...]
    point: AbstractExpression | tuple[AbstractExpression, ...]
    argument: Function | Conjugate | tuple
    scope = ("expression", "variable", "argument")

    def bound(self):
        entries = self.members(self.argument)
        return [
            *self.members(self.variable),
            *(m.operand if isinstance(m, Conjugate) else m for m in entries),
        ]

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def argument_spaces(self):
        return (*self.expression.argument_spaces, entry_space(self.argument))

    @cached_property
    def free_indices(self):
        edges = (self.expression, *self.members(self.point))
        return tuple(dict(p for x in edges for p in x.free_indices).items())

    @cached_property
    def scalar_type(self):
        if self.joined_scalar_type(self.variable) == "real":
            return self.expression.scalar_type
        return "complex"

    @cached_property
    def support(self):
        return self.expression.support
