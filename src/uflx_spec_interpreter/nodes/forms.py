from dataclasses import dataclass
from functools import cached_property

from .algebra import Conjugate
from .base import AbstractExpression, Base


@dataclass(frozen=True, eq=False)
class Space(Base):
    identifier = "toy.Space"
    name: str
    dual: bool = False
    value_shape: tuple = ()
    domain: object = None
    complex: bool = True
    opaque = ("name", "dual", "value_shape", "domain", "complex")


@dataclass(frozen=True, eq=False)
class Function(AbstractExpression):
    identifier = "uflx.Function"
    space: Space
    label: object
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


@dataclass(frozen=True, eq=False)
class Form(AbstractExpression):
    identifier = "uflx.Form"
    expression: Base
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


@dataclass(frozen=True, eq=False)
class Action(AbstractExpression):
    identifier = "uflx.Action"
    form: Base
    position: int
    argument: object
    opaque = ("position",)

    @cached_property
    def value_shape(self):
        return self.form.value_shape

    @cached_property
    def arity(self):
        return self.form.arity - 1


@dataclass(frozen=True, eq=False)
class GateauxDerivative(AbstractExpression):
    identifier = "uflx.GateauxDerivative"
    expression: Base
    variable: Base
    point: Base
    argument: Base
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
