import hashlib
import numbers
import weakref
from dataclasses import dataclass, fields

# Every node by its digest, so that equal nodes are the same object.
TABLE = weakref.WeakValueDictionary()


def encode(value):
    # A canonical encoding, tagged by type, so 1, 1.0 and True differ, and so do 0.0 and -0.0.
    match value:
        case Node():
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


class Node(metaclass=Interned):
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


@dataclass(frozen=True, eq=False)
class Space(Node):
    identifier = "toy.Space"
    name: str
    dual: bool = False
    shape: tuple = ()
    domain: object = None
    complex: bool = True
    opaque = ("name", "dual", "shape", "domain", "complex")


@dataclass(frozen=True, eq=False)
class Index(Node):
    identifier = "uflx.Index"
    label: object
    opaque = ("label",)
    labels = ("label",)


@dataclass(frozen=True, eq=False)
class Fn(Node):
    identifier = "uflx.Function"
    space: Space
    label: object
    opaque = ("space", "label")
    labels = ("label",)


@dataclass(frozen=True, eq=False)
class Num(Node):
    identifier = "toy.Num"
    value: complex
    opaque = ("value",)


@dataclass(frozen=True, eq=False)
class Zero(Node):
    identifier = "uflx.Zero"
    shape: tuple
    free: tuple
    args: tuple
    opaque = ("shape", "free", "args")


@dataclass(frozen=True, eq=False)
class Sum(Node):
    identifier = "uflx.Sum"
    a: Node
    b: Node


@dataclass(frozen=True, eq=False)
class Prod(Node):
    identifier = "uflx.Product"
    a: Node
    b: Node


@dataclass(frozen=True, eq=False)
class Div(Node):
    identifier = "uflx.Division"
    a: Node
    c: Node


@dataclass(frozen=True, eq=False)
class Conj(Node):
    identifier = "uflx.Conjugate"
    a: Node


@dataclass(frozen=True, eq=False)
class Re(Node):
    identifier = "uflx.RealPart"
    a: Node


@dataclass(frozen=True, eq=False)
class Im(Node):
    identifier = "uflx.ImaginaryPart"
    a: Node


@dataclass(frozen=True, eq=False)
class Abs(Node):
    identifier = "uflx.Absolute"
    a: Node


@dataclass(frozen=True, eq=False)
class Sqrt(Node):
    identifier = "uflx.SquareRoot"
    a: Node


@dataclass(frozen=True, eq=False)
class Log(Node):
    identifier = "uflx.Logarithm"
    a: Node


@dataclass(frozen=True, eq=False)
class Inv(Node):
    identifier = "uflx.Inverse"
    a: Node


@dataclass(frozen=True, eq=False)
class Power(Node):
    identifier = "uflx.Power"
    a: Node
    c: Node


@dataclass(frozen=True, eq=False)
class Tensor(Node):
    identifier = "uflx.Tensor"
    entries: tuple


@dataclass(frozen=True, eq=False)
class Indexed(Node):
    identifier = "uflx.Indexed"
    e: Node
    idx: tuple


@dataclass(frozen=True, eq=False)
class CT(Node):
    identifier = "uflx.ComponentTensor"
    e: Node
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)


@dataclass(frozen=True, eq=False)
class ISum(Node):
    identifier = "uflx.IndexSum"
    e: Node
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)


@dataclass(frozen=True, eq=False)
class Form(Node):
    identifier = "uflx.Form"
    e: Node
    args: tuple
    scope = ("e", "args")

    def bound(self):
        return [n for a in self.args for n in entry_fns(a)]


@dataclass(frozen=True, eq=False)
class Action(Node):
    identifier = "uflx.Action"
    f: Node
    p: int
    g: object
    opaque = ("p",)


@dataclass(frozen=True, eq=False)
class D(Node):
    identifier = "uflx.DirectionalDerivative"
    f: Node
    x: Node
    a: Node
    h: Node
    scope = ("f", "x")

    def bound(self):
        return [self.x]


def members(entry):
    return list(entry) if isinstance(entry, tuple) else [entry]


def entry_fns(entry):
    return [m.a if isinstance(m, Conj) else m for m in members(entry)]


@dataclass(frozen=True, eq=False)
class Coordinate(Node):
    identifier = "uflx.Coordinate"
    P: Node
    opaque = ("P",)


@dataclass(frozen=True, eq=False)
class Gradient(Node):
    identifier = "uflx.Gradient"
    b: Node
    P: Node
    opaque = ("P",)


@dataclass(frozen=True, eq=False)
class PullBack(Node):
    identifier = "uflx.PullBack"
    b: Node
    m: Node


@dataclass(frozen=True, eq=False)
class Integral(Node):
    identifier = "uflx.Integral"
    b: Node
    P: Node
    opaque = ("P",)


@dataclass(frozen=True, eq=False)
class Normal(Node):
    identifier = "uflx.Normal"
    D: Node
    E: Node
    side: Node


@dataclass(frozen=True, eq=False)
class Gateaux(Node):
    identifier = "uflx.GateauxDerivative"
    F: Node
    x: Node
    a: Node
    entry: Node
    scope = ("F", "x", "entry")

    def bound(self):
        return [self.x, self.entry.a if isinstance(self.entry, Conj) else self.entry]
