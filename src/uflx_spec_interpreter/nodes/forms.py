from dataclasses import dataclass
from functools import cached_property

from .algebra import Conjugate
from .base import AbstractExpression, Base


@dataclass(frozen=True, eq=False)
class Space(Base):
    identifier = "toy.Space"
    name: str
    dual: bool = False
    shape: tuple = ()
    domain: object = None
    complex: bool = True
    opaque = ("name", "dual", "shape", "domain", "complex")


@dataclass(frozen=True, eq=False)
class Function(AbstractExpression):
    identifier = "uflx.Function"
    space: Space
    label: object
    opaque = ("space", "label")
    labels = ("label",)

    @cached_property
    def value_shape(self):
        return () if self.space.dual else self.space.shape

    @cached_property
    def arity(self):
        return 1 if self.space.dual else 0

    @cached_property
    def support(self):
        return self.space.domain


@dataclass(frozen=True, eq=False)
class Form(AbstractExpression):
    identifier = "uflx.Form"
    e: Base
    args: tuple
    scope = ("e", "args")

    def bound(self):
        return [n for a in self.args for n in entry_fns(a)]

    @cached_property
    def value_shape(self):
        return self.e.value_shape

    @cached_property
    def arity(self):
        # The arguments of the expression come first, then the entries.
        return self.e.arity + len(self.args)


@dataclass(frozen=True, eq=False)
class Action(AbstractExpression):
    identifier = "uflx.Action"
    f: Base
    p: int
    g: object
    opaque = ("p",)

    @cached_property
    def value_shape(self):
        return self.f.value_shape

    @cached_property
    def arity(self):
        return self.f.arity - 1


def members(entry):
    return list(entry) if isinstance(entry, tuple) else [entry]


def entry_fns(entry):
    return [m.a if isinstance(m, Conjugate) else m for m in members(entry)]


@dataclass(frozen=True, eq=False)
class GateauxDerivative(AbstractExpression):
    identifier = "uflx.GateauxDerivative"
    F: Base
    x: Base
    a: Base
    entry: Base
    scope = ("F", "x", "entry")

    def bound(self):
        return [self.x, self.entry.a if isinstance(self.entry, Conjugate) else self.entry]

    @cached_property
    def value_shape(self):
        return self.F.value_shape

    @cached_property
    def arity(self):
        return self.F.arity + 1
