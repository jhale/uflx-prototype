from dataclasses import dataclass


class Node:
    # Fields that are data, not operands, so traversal skips them.
    opaque: tuple = ()
    # Fields whose nodes identify the node, which a substitution rebuilds.
    labels: tuple = ()
    # A binding node names the fields where its bound names are in scope.
    scope: tuple = ()

    def bound(self):
        return []


@dataclass(frozen=True)
class Space(Node):
    name: str
    dual: bool = False
    shape: tuple = ()
    domain: object = None
    complex: bool = True
    opaque = ("name", "dual", "shape", "domain", "complex")


@dataclass(frozen=True)
class Index(Node):
    label: object
    opaque = ("label",)
    labels = ("label",)


@dataclass(frozen=True)
class Fn(Node):
    space: Space
    label: object
    opaque = ("space", "label")
    labels = ("label",)


@dataclass(frozen=True)
class Num(Node):
    value: complex
    opaque = ("value",)


@dataclass(frozen=True)
class Zero(Node):
    shape: tuple
    free: tuple
    args: tuple
    opaque = ("shape", "free", "args")


@dataclass(frozen=True)
class Sum(Node):
    a: Node
    b: Node


@dataclass(frozen=True)
class Prod(Node):
    a: Node
    b: Node


@dataclass(frozen=True)
class Div(Node):
    a: Node
    c: Node


@dataclass(frozen=True)
class Conj(Node):
    a: Node


@dataclass(frozen=True)
class Re(Node):
    a: Node


@dataclass(frozen=True)
class Im(Node):
    a: Node


@dataclass(frozen=True)
class Abs(Node):
    a: Node


@dataclass(frozen=True)
class Sqrt(Node):
    a: Node


@dataclass(frozen=True)
class Log(Node):
    a: Node


@dataclass(frozen=True)
class Inv(Node):
    a: Node


@dataclass(frozen=True)
class Power(Node):
    a: Node
    c: Node


@dataclass(frozen=True)
class Tensor(Node):
    entries: tuple


@dataclass(frozen=True)
class Indexed(Node):
    e: Node
    idx: tuple


@dataclass(frozen=True)
class CT(Node):
    e: Node
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)


@dataclass(frozen=True)
class ISum(Node):
    e: Node
    idx: tuple
    scope = ("e", "idx")

    def bound(self):
        return list(self.idx)


@dataclass(frozen=True)
class Form(Node):
    e: Node
    args: tuple
    scope = ("e", "args")

    def bound(self):
        return [n for a in self.args for n in entry_fns(a)]


@dataclass(frozen=True)
class Action(Node):
    f: Node
    p: int
    g: object
    opaque = ("p",)


@dataclass(frozen=True)
class D(Node):
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


@dataclass(frozen=True)
class Coordinate(Node):
    P: Node
    opaque = ("P",)


@dataclass(frozen=True)
class Gradient(Node):
    b: Node
    P: Node
    opaque = ("P",)


@dataclass(frozen=True)
class PullBack(Node):
    b: Node
    m: Node


@dataclass(frozen=True)
class Integral(Node):
    b: Node
    P: Node
    opaque = ("P",)


@dataclass(frozen=True)
class Normal(Node):
    D: Node
    E: Node
    side: Node


@dataclass(frozen=True)
class Gateaux(Node):
    F: Node
    x: Node
    a: Node
    entry: Node
    scope = ("F", "x", "entry")

    def bound(self):
        return [self.x, self.entry.a if isinstance(self.entry, Conj) else self.entry]
