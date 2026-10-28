from dataclasses import dataclass, fields, replace
from functools import lru_cache


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


def operands(e, names=None):
    # The nodes on the edges of e, in order, through tuples.
    out = []

    def walk(x):
        if isinstance(x, Node):
            out.append(x)
        elif isinstance(x, tuple):
            for y in x:
                walk(y)

    names = [f.name for f in fields(e) if f.name not in e.opaque] if names is None else names
    for n in names:
        walk(getattr(e, n))
    return out


def map_fields(e, fn, names=None):
    # Rebuild e with fn applied to every node in the named fields.
    def walk(x):
        if isinstance(x, Node):
            return fn(x)
        if isinstance(x, tuple):
            return tuple(walk(y) for y in x)
        return x

    names = [f.name for f in fields(e) if f.name not in e.opaque] if names is None else names
    return replace(e, **{n: walk(getattr(e, n)) for n in names})


def occurs(r, e):
    # Whether r is free in e.
    if e == r:
        return True
    names = [f.name for f in fields(e) if f.name not in e.opaque]
    if any(occurs(n, r) for n in e.bound()):
        names = [n for n in names if n not in e.scope]
    return any(occurs(r, x) for x in operands(e, names))


def rename(node, name, avoid):
    # The label refers to the bound operands and to avoid, never to the binding node.
    body = tuple(getattr(node, f) for f in node.scope)
    label = ("uflx.rename", name, body, tuple(avoid))
    if isinstance(name, Index):
        return Index(label)
    if isinstance(name, Fn):
        return Fn(name.space, label)
    raise TypeError("a compound variable has no new name")


def freshen(e, avoid):
    # Rename the bound names of e that are free in an expression of avoid.
    for n in e.bound():
        if any(occurs(n, s) for s in avoid):
            m = rename(e, n, avoid)
            e = map_fields(e, lambda x: substitute(x, {n: m}), e.scope)
    return e


def substitute(e, mapping):
    # Simultaneous capture-avoiding substitution of mapping[r] for each r in e.
    return _substitute(e, tuple(mapping.items()))


@lru_cache(maxsize=None)
def _substitute(e, items):
    mapping = dict(items)
    if e in mapping:
        return mapping[e]
    if e.labels:
        # A derived name whose label contains a replaced object is rebuilt, and a renamed name is not.
        renamed = isinstance(e.label, tuple) and e.label[:1] == ("uflx.rename",)
        return e if renamed else map_fields(e, lambda x: _substitute(x, items), e.labels)
    if not e.scope:
        return map_fields(e, lambda x: _substitute(x, items))
    inner = {r: s for r, s in mapping.items() if not any(occurs(n, r) for n in e.bound())}
    fresh = freshen(e, list(inner.values()))
    if not fresh.scope:
        return _substitute(fresh, items)
    e = fresh
    outside = [f.name for f in fields(e) if f.name not in e.opaque and f.name not in e.scope]
    e = map_fields(e, lambda x: _substitute(x, items), outside)
    return map_fields(e, lambda x: _substitute(x, tuple(inner.items())), e.scope)


# Domains and fields, for the integrals pass.
@dataclass(frozen=True)
class CoordDomain(Node):
    name: str
    dim: int
    opaque = ("name", "dim")


@dataclass(frozen=True)
class DomainMap(Node):
    P: Node
    psi: Node
    Q: Node


@dataclass(frozen=True)
class Decomposed(Node):
    pieces: tuple


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


def ambient_dim(P):
    if isinstance(P, CoordDomain):
        return P.dim
    return ambient_dim(P.pieces[0].Q)


def topdim(P):
    return P.dim if isinstance(P, CoordDomain) else topdim(P.pieces[0].P)


def ambient(P):
    return P.pieces[0].Q if isinstance(P, Decomposed) else P


@dataclass(frozen=True)
class Gathered(Node):
    # An index read from a table at the given entries.
    table: Node
    indices: tuple


@dataclass(frozen=True)
class CellMap(Node):
    # Piece j of D seen as a domain in D, for the cell selected by entry (an Index or a Gathered).
    D: Node
    j: int
    entry: Node
    opaque = ("j",)

    @property
    def P(self):
        return self.D.pieces[self.j].P

    @property
    def Q(self):
        return self.D

    @property
    def psi(self):
        piece = self.D.pieces[self.j]
        family = [i for i in operands_free(piece.psi)]
        return substitute(piece.psi, {family[0]: self.entry}) if family and family[0] != self.entry else piece.psi


@dataclass(frozen=True)
class SideMap(Node):
    # The map from the cells of E into D, from one side, through the side tables.
    D: Node
    E: Node
    cells: Node
    positions: Node


@dataclass(frozen=True)
class Normal(Node):
    D: Node
    E: Node
    side: Node


def operands_free(psi):
    from static import free
    return list(free(psi))


def cell_domain(D, j):
    from static import free
    family = list(free(D.pieces[j].psi))
    return CellMap(D, j, family[0]) if family else DomainMap(D.pieces[j].P, D.pieces[j].psi, D)


# The commuting node classes and the operands into which an evaluated node moves.
COMMUTING = {
    Sum: ("a", "b"), Tensor: ("entries",), Indexed: ("e",), CT: ("e",), ISum: ("e",),
    Form: ("e",), D: ("f",), Conj: ("a",), Re: ("a",), Im: ("a",), Div: ("a",),
    Gradient: ("b",), PullBack: ("b",), Integral: ("b",),
}


def commute(f, evaluate, avoid):
    # Move an evaluated node into the listed operands, after making a binding node fresh.
    fresh = freshen(f, avoid) if f.scope else f
    return map_fields(fresh, evaluate, COMMUTING[type(fresh)])


@dataclass(frozen=True)
class Gateaux(Node):
    F: Node
    x: Node
    a: Node
    entry: Node
    scope = ("F", "x", "entry")

    def bound(self):
        return [self.x, self.entry.a if isinstance(self.entry, Conj) else self.entry]
