import weakref

from .names import is_renamed_label, renamed_label
from .nodes import AbstractExpression, Base, Function, Index


def operands(e, names=None):
    # The nodes on the edges of e, in order, through tuples, or those of the named arguments.
    return list(edges(e) if names is None else nodes_in(e, names))


def edges(e):
    # All construction dependencies, including spaces, maps, labels and literal parts.
    memo = vars(e)
    if "edges" not in memo:
        memo["edges"] = nodes_in(e, e.argument_names)
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
    from .nodes import ConjugateSpace, DualSpace
    from .spaces import conjugate_space, dual_space

    if isinstance(e, ConjugateSpace):
        return conjugate_space(values["space"])
    if isinstance(e, DualSpace):
        return dual_space(values["space"])
    return type(e)(*values.values())


def expressions(e):
    # The operands of e that are expressions, which a grammar restricts.
    return [x for x in nodes_in(e, lowered(e)) if isinstance(x, AbstractExpression)]


def map_fields(e, fn, names=None):
    # Rebuild e with fn applied to every node in the named fields.
    def walk(x):
        if isinstance(x, Base):
            return fn(x)
        if isinstance(x, tuple):
            return tuple(walk(y) for y in x)
        return x

    names = e.argument_names if names is None else names
    return rebuild(e, **{n: walk(getattr(e, n)) for n in names})


def occurs(r, e):
    # Whether r is free in e, kept on e and weakly in r so that each is freed with its node.
    if not isinstance(e, Base):
        return False
    memo = vars(e).setdefault("occurrences", weakref.WeakKeyDictionary())
    if r not in memo:
        memo[r] = e == r or any(occurs(r, x) for x in operands(e, scoped(r, e)))
    return memo[r]


def scoped(r, e):
    # The arguments of e in which r can be free, without the scope if r names a bound name.
    names = list(e.argument_names) if isinstance(r, Index) else lowered(e)
    names = [n for n in names if n not in e.labels]
    if any(occurs(n, r) for n in e.bound()):
        names = [n for n in names if n not in e.scope]
    return names


def rename(node, name, avoid):
    # The label refers to the bound operands and to avoid, never to the binding node.
    label = renamed_label(node, name, avoid)
    if isinstance(name, Index):
        return type(name)(label)
    if isinstance(name, Function):
        return Function(name.space, label)
    raise TypeError("a compound variable has no new name")


def freshen(e, avoid):
    # Rename the bound names of e that are free in an expression of avoid.
    renames = {n: rename(e, n, avoid) for n in e.bound() if any(occurs(n, s) for s in avoid)}
    return map_fields(e, lambda x: substitute(x, renames), e.scope) if renames else e


def substitute(e, mapping):
    # Simultaneous capture-avoiding substitution of mapping[r] for each r in e.
    return memoized(substituted)(e, tuple(sorted(mapping.items(), key=lambda item: item[0].digest)))


def substituted(walk, e, items, dependencies=True):
    mapping = dict(items)
    if e in mapping:
        return mapping[e]
    from .nodes import GatheredIndex, Zero
    from .validation import merge_indices

    if isinstance(e, Zero):
        # Zero's free-index signature changes when an index is selected or gathered.
        pairs = []
        for i, extent in e.free_indices:
            entry = walk(i, items)
            size = walk(extent, items) if isinstance(extent, Base) else extent
            if isinstance(entry, Index):
                pairs.append(((entry, size),))
            elif isinstance(entry, GatheredIndex):
                if entry.extent != size:
                    raise IndexError("replacement gather has the wrong extent")
                pairs.append(entry.free_indices)
            elif type(entry) is not int:
                raise TypeError("replacement of an index must be an index entry")
            elif type(size) is not int or not 0 <= entry < size:
                raise IndexError("replacement index is outside the axis")
        new = map_fields(
            e, lambda x: walk(x, items), ("value_shape", "argument_spaces", "scalar_type")
        )
        return rebuild(new, free_indices=merge_indices(*pairs))
    if e.labels:
        if not dependencies:
            return e
        # A derived name with a replaced object in its label is rebuilt, and a renamed name is not.
        if is_renamed_label(e.label):
            return map_fields(
                e, lambda x: walk(x, items), [n for n in e.argument_names if n not in e.labels]
            )
        return map_fields(
            e,
            lambda x: walk(x, items),
            e.argument_names,
        )
    if not e.scope:
        return map_fields(
            e,
            lambda x: walk(x, items),
            e.argument_names if dependencies else (*e.value_fields, *e.index_fields),
        )
    inner = {r: s for r, s in mapping.items() if not any(occurs(n, r) for n in e.bound())}
    e = freshen(e, list(inner.values()))
    outside = [
        n
        for n in (e.argument_names if dependencies else (*e.value_fields, *e.index_fields))
        if n not in e.scope
    ]
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


def lowered(e):
    # Only explicitly declared value operands are evaluated.
    return e.value_fields


def bottom_up(f, rewrite):
    # Rewrite each node after its operands, once for each node of the graph of f.
    return memoized(lambda walk, e: rewrite(map_fields(e, walk, lowered(e))))(f)


def everywhere(e, holds):
    # Whether holds is true at every node of the graph of e.
    return memoized(lambda walk, n: holds(n) and all(map(walk, expressions(n))))(e)


def replace(e, mapping):
    """Substitute simultaneously through construction dependencies, respecting binding."""
    if not isinstance(e, Base) or type(mapping) is not dict:
        raise TypeError("replace takes a graph object and a dictionary")
    if not all(isinstance(key, Base) for key in mapping):
        raise TypeError("replacement keys must be graph objects")
    return substitute(e, mapping)


def substitute_values(e, mapping):
    # Value instantiation holds geometry and identity metadata fixed.
    items = tuple(sorted(mapping.items(), key=lambda item: item[0].digest))
    return memoized(lambda walk, node, pairs: substituted(walk, node, pairs, False))(e, items)
