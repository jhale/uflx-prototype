import hashlib
import numbers
import weakref
from dataclasses import dataclass, fields
from functools import cached_property

# Every node by its digest, so that equal nodes are the same object.
TABLE = weakref.WeakValueDictionary()


def encode(value):
    # A canonical encoding, tagged by type, so 1, 1.0 and True differ, and so do 0.0 and -0.0.
    match value:
        case Base():
            return b"N" + value.digest.to_bytes(16, "big")
        case None:
            return b"n"
        case bool():
            return b"b1" if value else b"b0"
        case numbers.Integral():
            return b"i" + str(int(value)).encode() + b";"
        case numbers.Real():
            return b"f" + float(value).hex().encode() + b";"
        case numbers.Complex():
            return b"c" + encode(float(value.real)) + encode(float(value.imag))
        case str():
            data = value.encode()
            return b"s" + str(len(data)).encode() + b":" + data
        case tuple():
            return b"t" + str(len(value)).encode() + b":" + b"".join(map(encode, value))
    raise TypeError(f"not a construction argument: {value!r}")


def digest_of(identifier, values):
    # Seed-free, unlike the built-in hash of str and bytes, so equal in every process.
    data = encode(identifier) + encode(tuple(values))
    return int.from_bytes(hashlib.blake2b(data, digest_size=16).digest(), "big")


class Interned(type):
    def __call__(cls, *args, **kwargs):
        if "identifier" not in cls.__dict__:
            raise TypeError(f"{cls.__name__} states no identifier")
        node = super().__call__(*args, **kwargs)
        digest = digest_of(cls.identifier, (getattr(node, f.name) for f in fields(node)))
        existing = TABLE.get(digest)
        if existing is not None:
            return existing
        object.__setattr__(node, "digest", digest)
        TABLE[digest] = node
        return node


class Base(metaclass=Interned):
    # Fields that are data, not operands, so traversal skips them.
    opaque: tuple = ()
    # Fields whose nodes identify the node, which a substitution rebuilds.
    labels: tuple = ()
    # A binding node names the fields where its bound names are in scope.
    scope: tuple = ()

    def bound(self):
        return []

    def __hash__(self):
        return self.digest % (2**61 - 1)


class AbstractExpression(Base):
    # The static type, derived from the construction arguments once for each interned node.
    # The rules import this module, so they are imported where they are used.
    @cached_property
    def value_shape(self):
        from .static import rule_shape

        return rule_shape(self)

    @cached_property
    def free_indices(self):
        from .static import rule_free

        return tuple(rule_free(self).items())

    @cached_property
    def arity(self):
        from .static import rule_arity

        return rule_arity(self)

    @cached_property
    def support(self):
        from .static import rule_support

        return rule_support(self)


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
class Index(Base):
    identifier = "uflx.Index"
    label: object
    opaque = ("label",)
    labels = ("label",)


@dataclass(frozen=True, eq=False)
class Fn(AbstractExpression):
    identifier = "uflx.Function"
    space: Space
    label: object
    opaque = ("space", "label")
    labels = ("label",)


@dataclass(frozen=True, eq=False)
class Num(AbstractExpression):
    identifier = "toy.Num"
    value: complex
    opaque = ("value",)


@dataclass(frozen=True, eq=False)
class Zero(AbstractExpression):
    identifier = "uflx.Zero"
    shape: tuple
    free: tuple
    args: tuple
    opaque = ("shape", "free", "args")


@dataclass(frozen=True, eq=False)
class Sum(AbstractExpression):
    identifier = "uflx.Sum"
    a: Base
    b: Base


@dataclass(frozen=True, eq=False)
class Prod(AbstractExpression):
    identifier = "uflx.Product"
    a: Base
    b: Base


@dataclass(frozen=True, eq=False)
class Div(AbstractExpression):
    identifier = "uflx.Division"
    a: Base
    c: Base


@dataclass(frozen=True, eq=False)
class Conj(AbstractExpression):
    identifier = "uflx.Conjugate"
    a: Base


@dataclass(frozen=True, eq=False)
class Re(AbstractExpression):
    identifier = "uflx.RealPart"
    a: Base


@dataclass(frozen=True, eq=False)
class Im(AbstractExpression):
    identifier = "uflx.ImaginaryPart"
    a: Base


@dataclass(frozen=True, eq=False)
class Abs(AbstractExpression):
    identifier = "uflx.Absolute"
    a: Base


@dataclass(frozen=True, eq=False)
class Sqrt(AbstractExpression):
    identifier = "uflx.SquareRoot"
    a: Base


@dataclass(frozen=True, eq=False)
class Log(AbstractExpression):
    identifier = "uflx.Logarithm"
    a: Base


@dataclass(frozen=True, eq=False)
class Inv(AbstractExpression):
    identifier = "uflx.Inverse"
    a: Base


@dataclass(frozen=True, eq=False)
class Power(AbstractExpression):
    identifier = "uflx.Power"
    a: Base
    c: Base


@dataclass(frozen=True, eq=False)
class Tensor(AbstractExpression):
    identifier = "uflx.Tensor"
    entries: tuple


@dataclass(frozen=True, eq=False)
class Indexed(AbstractExpression):
    identifier = "uflx.Indexed"
    e: Base
    idx: tuple


@dataclass(frozen=True, eq=False)
class CT(AbstractExpression):
    identifier = "uflx.ComponentTensor"
    e: Base
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)


@dataclass(frozen=True, eq=False)
class ISum(AbstractExpression):
    identifier = "uflx.IndexSum"
    e: Base
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)


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


@dataclass(frozen=True, eq=False)
class D(AbstractExpression):
    identifier = "uflx.DirectionalDerivative"
    f: Base
    x: Base
    a: Base
    h: Base
    scope = ("f", "x")

    def bound(self):
        return [self.x]


def members(entry):
    return list(entry) if isinstance(entry, tuple) else [entry]


def entry_fns(entry):
    return [m.a if isinstance(m, Conj) else m for m in members(entry)]


@dataclass(frozen=True, eq=False)
class Coordinate(AbstractExpression):
    identifier = "uflx.Coordinate"
    P: Base
    opaque = ("P",)


@dataclass(frozen=True, eq=False)
class Gradient(AbstractExpression):
    identifier = "uflx.Gradient"
    b: Base
    P: Base
    opaque = ("P",)


@dataclass(frozen=True, eq=False)
class PullBack(AbstractExpression):
    identifier = "uflx.PullBack"
    b: Base
    m: Base


@dataclass(frozen=True, eq=False)
class Integral(AbstractExpression):
    identifier = "uflx.Integral"
    b: Base
    P: Base
    opaque = ("P",)


@dataclass(frozen=True, eq=False)
class Normal(AbstractExpression):
    identifier = "uflx.Normal"
    D: Base
    E: Base
    side: Base


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
