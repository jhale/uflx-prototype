from uflx_spec_interpreter._src.nodes import (
    AbstractExpression,
    Action,
    ComponentTensor,
    Conjugate,
    ConjugateSpace,
    DirectionalDerivative,
    Division,
    DualSpace,
    Form,
    Function,
    GateauxDerivative,
    Gradient,
    Indexed,
    IndexSum,
    Integral,
    Product,
    PullBack,
    Sum,
    Tensor,
)
from uflx_spec_interpreter._src.substitution import nodes_in, operands, traversed


def contains(e, predicate):
    # Memoised, since the expressions share subexpressions.
    seen = {}

    def walk(n):
        if n not in seen:
            seen[n] = predicate(n) or any(walk(c) for c in operands(n))
        return seen[n]

    return walk(e)


def assert_clean(counts):
    for key in ("value", "static", "grammar", "typed"):
        assert counts.get(key, 0) == 0, counts
    if "lazy" in counts:
        assert counts["rejected"] == counts["lazy"] > 0, counts


def pairing(action):
    return isinstance(action.form, Function) and isinstance(action.form.space, DualSpace)


def static_type(e):
    # The free indices compare as a dict, since a pass may reorder them.
    return e.value_shape, dict(e.free_indices), e.argument_spaces


# The operands whose argument spaces a node passes on, by the static type table.
PASSED = {
    Sum: ("first", "second"),
    Product: ("first", "second"),
    Division: ("first",),
    Conjugate: ("operand",),
    Tensor: ("entries",),
    Indexed: ("expression",),
    ComponentTensor: ("expression",),
    IndexSum: ("expression",),
    Gradient: ("expression",),
    DirectionalDerivative: ("expression",),
    GateauxDerivative: ("expression",),
    Integral: ("expression",),
    PullBack: ("expression",),
    Form: ("expression",),
    Action: ("form", "argument"),
}


def ill_typed(e):
    # The first node that breaks a rule of the argument spaces, which construction would raise.
    # The message names the spaces only, since the repr of a shared graph can be exponential.
    seen = {}

    def walk(n):
        if n not in seen:
            seen[n] = node_error(n) or next(filter(None, map(walk, operands(n))), None)
        return seen[n]

    return walk(e)


def node_error(n):
    passed = PASSED.get(type(n), ())
    others = [m for name in traversed(n) if name not in passed for m in nodes_in(n, [name])]
    for m in others:
        if isinstance(m, AbstractExpression) and m.argument_spaces:
            return f"{type(n).__name__} has an operand with argument spaces {m.argument_spaces}"
    match n:
        case Sum(a, b) if a.argument_spaces != b.argument_spaces:
            return f"Sum of {a.argument_spaces} and {b.argument_spaces}"
        case Tensor(entries) if len({x.argument_spaces for x in entries}) > 1:
            return f"Tensor of {[x.argument_spaces for x in entries]}"
        case Action(f, p, g):
            if p >= len(f.argument_spaces):
                return f"Action at {p} of {f.argument_spaces}"
            if spaces_of_value(f.argument_spaces[p]) != spaces_of(g):
                return f"Action at {p} of {f.argument_spaces} on {spaces_of(g)}"
    return None


def spaces_of_value(V):
    # The argument spaces of a value for an argument in V, or in the conjugate of V.
    if isinstance(V, tuple):
        return tuple(map(spaces_of_value, V))
    return (V.space if isinstance(V, ConjugateSpace) else V).argument_spaces


def spaces_of(g):
    return tuple(map(spaces_of, g)) if isinstance(g, tuple) else g.argument_spaces


def check_types(bad, *es):
    for e in es:
        error = ill_typed(e)
        if error:
            bad["typed"] = bad.get("typed", 0) + 1
            if bad["typed"] <= 2:
                print("ill typed:", error)
            return
