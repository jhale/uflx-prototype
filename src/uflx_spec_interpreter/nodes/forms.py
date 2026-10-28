from dataclasses import dataclass

from .algebra import Conj
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
class Fn(AbstractExpression):
    identifier = "uflx.Function"
    space: Space
    label: object
    opaque = ("space", "label")
    labels = ("label",)


@dataclass(frozen=True, eq=False)
class Form(AbstractExpression):
    identifier = "uflx.Form"
    e: Base
    args: tuple
    scope = ("e", "args")

    def bound(self):
        return [n for a in self.args for n in entry_fns(a)]


@dataclass(frozen=True, eq=False)
class Action(AbstractExpression):
    identifier = "uflx.Action"
    f: Base
    p: int
    g: object
    opaque = ("p",)


def members(entry):
    return list(entry) if isinstance(entry, tuple) else [entry]


def entry_fns(entry):
    return [m.a if isinstance(m, Conj) else m for m in members(entry)]


@dataclass(frozen=True, eq=False)
class Gateaux(AbstractExpression):
    identifier = "uflx.GateauxDerivative"
    F: Base
    x: Base
    a: Base
    entry: Base
    scope = ("F", "x", "entry")

    def bound(self):
        return [self.x, self.entry.a if isinstance(self.entry, Conj) else self.entry]
