import weakref

from .nodes import AbstractExpression, Base, EllipsisIndex, Function, Index


def operands(e, names=None):
    # The nodes on the edges of e, in order, through tuples, or those of the named arguments.
    return list(edges(e) if names is None else nodes_in(e, names))


def edges(e):
    # The edges of rule 5, without the opaque arguments, kept on the node like its static type.
    memo = vars(e)
    if "edges" not in memo:
        memo["edges"] = nodes_in(e, traversed(e))
    return memo["edges"]


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
    # Whether r is free in e, kept on e and weakly in r so that each is freed with its node.
    memo = vars(e).setdefault("occurrences", weakref.WeakKeyDictionary())
    if r not in memo:
        memo[r] = e == r or any(occurs(r, x) for x in operands(e, scoped(r, e)))
    return memo[r]


def scoped(r, e):
    # The arguments of e in which r can be free, without the scope if r names a bound name.
    names = traversed(e)
    if any(occurs(n, r) for n in e.bound()):
        names = [n for n in names if n not in e.scope]
    return names


def rename(node, name, avoid):
    # The label refers to the bound operands and to avoid, never to the binding node.
    body = tuple(getattr(node, f) for f in node.scope)
    label = ("uflx.rename", name, body, tuple(avoid))
    if isinstance(name, Index | EllipsisIndex):
        return type(name)(label)
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
    return memoized(substituted)(e, tuple(mapping.items()))


def substituted(walk, e, items):
    mapping = dict(items)
    if e in mapping:
        return mapping[e]
    if e.labels:
        # A derived name with a replaced object in its label is rebuilt, and a renamed name is not.
        renamed = isinstance(e.label, tuple) and e.label[:1] == ("uflx.rename",)
        return e if renamed else map_fields(e, lambda x: walk(x, items), e.labels)
    if not e.scope:
        return map_fields(e, lambda x: walk(x, items))
    inner = {r: s for r, s in mapping.items() if not any(occurs(n, r) for n in e.bound())}
    fresh = freshen(e, list(inner.values()))
    if not fresh.scope:
        return walk(fresh, items)
    e = fresh
    outside = [n for n in traversed(e) if n not in e.scope]
    e = map_fields(e, lambda x: walk(x, items), outside)
    return map_fields(e, lambda x: walk(x, tuple(inner.items())), e.scope)


def memoized(rule):
    # The recursion of rule through one memo, which lasts as long as the returned function.
    memo = {}

    def recurse(*args):
        if args not in memo:
            memo[args] = rule(recurse, *args)
        return memo[args]

    return recurse


def bottom_up(f, rewrite):
    # Rewrite each node after its operands, once for each node of the graph of f.
    return memoized(lambda walk, e: rewrite(map_fields(e, walk)))(f)


def everywhere(e, holds):
    # Whether holds is true at every node of the graph of e.
    return memoized(lambda walk, n: holds(n) and all(map(walk, expressions(n))))(e)
