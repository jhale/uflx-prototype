from __future__ import annotations

from functools import cached_property
from typing import TYPE_CHECKING, final

from .algebra import Conjugate
from .base import AbstractExpression, Base

if TYPE_CHECKING:
    from .domains import AbstractDomain


@final
class Space(Base):
    identifier = "toy.Space"
    name: str
    dual: bool
    value_shape: tuple[int, ...]
    domain: AbstractDomain | None
    complex: bool
    opaque = ("name", "dual", "value_shape", "domain", "complex")


@final
class Function(AbstractExpression):
    identifier = "uflx.Function"
    space: Space
    label: str | tuple
    opaque = ("space", "label")
    labels = ("label",)

    @cached_property
    def value_shape(self):
        return () if self.space.dual else self.space.value_shape

    @cached_property
    def arity(self):
        return 1 if self.space.dual else 0

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
    def arity(self):
        # The arguments of the expression come first, then the entries.
        return self.expression.arity + len(self.arguments)

    @cached_property
    def free_indices(self):
        return self.expression.free_indices

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
    def arity(self):
        return self.form.arity - 1

    @cached_property
    def free_indices(self):
        members = self.argument if isinstance(self.argument, tuple) else (self.argument,)
        return tuple(dict(p for x in (self.form, *members) for p in x.free_indices).items())

    @cached_property
    def support(self):
        return self.form.support


@final
class GateauxDerivative(AbstractExpression):
    identifier = "uflx.GateauxDerivative"
    expression: AbstractExpression
    variable: Function
    point: AbstractExpression
    argument: Function | Conjugate
    scope = ("expression", "variable", "argument")

    def bound(self):
        return [
            self.variable,
            self.argument.operand if isinstance(self.argument, Conjugate) else self.argument,
        ]

    @cached_property
    def value_shape(self):
        return self.expression.value_shape

    @cached_property
    def arity(self):
        return self.expression.arity + 1

    @cached_property
    def free_indices(self):
        return tuple(dict(self.expression.free_indices + self.point.free_indices).items())

    @cached_property
    def support(self):
        return self.expression.support
