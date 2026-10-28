from functools import cache

from .nodes import AbstractExpression, Base, Function, Index


def operands(e, names=None):
    # The nodes on the edges of e, in order, through tuples, or those of the named arguments.
    return list(edges(e) if names is None else nodes_in(e, names))


@cache
def edges(e):
    # The edges of rule 5, without the opaque arguments, computed once for each node.
    return nodes_in(e, traversed(e))


def nodes_in(e, names):
    out = []

    def walk(x):
        if isinstance(x, Base):
            out.append(x)
        elif isinstance(x, tuple):
            for y in x:
                walk(y)

    for name in names:
        walk(getattr(e, name))
    return tuple(out)


def rebuild(e, **changes):
    # An unknown name adds an argument, which the class rejects.
    values = dict(zip(e.argument_names, e.construction_arguments)) | changes
    return type(e)(*values.values())


def traversed(e):
    # The construction arguments that traversal visits.
    return [n for n in e.argument_names if n not in e.opaque]


def expressions(e):
    # The operands of e that are expressions, which a grammar restricts.
    return [x for x in operands(e) if isinstance(x, AbstractExpression)]


def map_fields(e, fn, names=None):
    # Rebuild e with fn applied to every node in the named fields.
    def walk(x):
        if isinstance(x, Base):
            return fn(x)
        if isinstance(x, tuple):
            return tuple(walk(y) for y in x)
        return x

    names = traversed(e) if names is None else names
    return rebuild(e, **{n: walk(getattr(e, n)) for n in names})


def occurs(r, e):
    # Whether r is free in e.
    if e == r:
        return True
    names = traversed(e)
    if any(occurs(n, r) for n in e.bound()):
        names = [n for n in names if n not in e.scope]
    return any(occurs(r, x) for x in operands(e, names))


def rename(node, name, avoid):
    # The label refers to the bound operands and to avoid, never to the binding node.
    body = tuple(getattr(node, f) for f in node.scope)
    label = ("uflx.rename", name, body, tuple(avoid))
    if isinstance(name, Index):
        return Index(label)
    if isinstance(name, Function):
        return Function(name.space, label)
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


@cache
def _substitute(e, items):
    mapping = dict(items)
    if e in mapping:
        return mapping[e]
    if e.labels:
        # A derived name with a replaced object in its label is rebuilt, and a renamed name is not.
        renamed = isinstance(e.label, tuple) and e.label[:1] == ("uflx.rename",)
        return e if renamed else map_fields(e, lambda x: _substitute(x, items), e.labels)
    if not e.scope:
        return map_fields(e, lambda x: _substitute(x, items))
    inner = {r: s for r, s in mapping.items() if not any(occurs(n, r) for n in e.bound())}
    fresh = freshen(e, list(inner.values()))
    if not fresh.scope:
        return _substitute(fresh, items)
    e = fresh
    outside = [n for n in traversed(e) if n not in e.scope]
    e = map_fields(e, lambda x: _substitute(x, items), outside)
    return map_fields(e, lambda x: _substitute(x, tuple(inner.items())), e.scope)
